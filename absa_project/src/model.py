"""
model.py — Dual-head BERT model for Aspect-Based Sentiment Analysis.

  Head 1 (NER head):  Token-level classification → aspect term extraction (B/I/O)
  Head 2 (Cat head):  Sentence-level multi-class → category × polarity prediction

The two heads share the same BERT encoder.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast, BertConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ABSAModel(nn.Module):
    """
    Dual-head BERT for Aspect-Based Sentiment Analysis.

    Forward output:
        ner_logits  (B, seq_len, num_ner_labels)
        cat_logits  (B, num_categories, num_polarities)
    """

    def __init__(self, config):
        super().__init__()
        self.config     = config
        self.bert       = BertModel.from_pretrained(config.pretrained_model_name)
        hidden_size     = self.bert.config.hidden_size          # 768

        # Task A — token-level NER
        self.ner_dropout = nn.Dropout(0.1)
        self.ner_head    = nn.Linear(hidden_size, config.num_ner_labels)

        # Task B — sentence-level category-sentiment
        # Uses CLS token representation, one output per (category, polarity) pair
        self.cat_dropout = nn.Dropout(0.1)
        self.cat_head    = nn.Linear(
            hidden_size,
            config.num_categories * config.num_polarities,
        )

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:      torch.Tensor,           # (B, L)
        attention_mask: torch.Tensor,           # (B, L)
        token_type_ids: Optional[torch.Tensor] = None,  # (B, L)
        ner_labels:     Optional[torch.Tensor] = None,  # (B, L)
        cat_labels:     Optional[torch.Tensor] = None,  # (B, num_categories)
    ) -> Dict[str, torch.Tensor]:

        outputs = self.bert(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            token_type_ids  = token_type_ids,
        )
        seq_out = outputs.last_hidden_state   # (B, L, H)
        cls_out = outputs.pooler_output       # (B, H)

        # Task A logits
        ner_logits = self.ner_head(self.ner_dropout(seq_out))   # (B, L, num_ner)

        # Task B logits → reshape to (B, num_cat, num_pol)
        cat_flat   = self.cat_head(self.cat_dropout(cls_out))   # (B, num_cat*num_pol)
        cat_logits = cat_flat.view(
            -1,
            self.config.num_categories,
            self.config.num_polarities,
        )                                                        # (B, num_cat, num_pol)

        result: Dict[str, torch.Tensor] = {
            "ner_logits": ner_logits,
            "cat_logits": cat_logits,
        }

        # ── Compute losses when labels are provided ────────────────────────
        if ner_labels is not None and cat_labels is not None:
            ner_loss = self._ner_loss(ner_logits, ner_labels)
            cat_loss = self._cat_loss(cat_logits, cat_labels)
            result["ner_loss"] = ner_loss
            result["cat_loss"] = cat_loss
            result["loss"]     = ner_loss + cat_loss

        return result

    # ──────────────────────────────────────────────────────────────────────
    def _ner_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on non-padding tokens (ignore -100)."""
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        # logits: (B, L, C) → (B*L, C);  labels: (B, L) → (B*L,)
        return loss_fn(
            logits.view(-1, self.config.num_ner_labels),
            labels.view(-1),
        )

    def _cat_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Per-category cross-entropy — fully vectorised.

        CHANGE 3: replace the Python `for c in range(num_cat)` loop with a single
        F.cross_entropy call over a flattened view.

            logits : (B, num_cat, num_pol)
            labels : (B, num_cat)          — value -1 means category absent

        We flatten to:
            logits_flat : (B * num_cat, num_pol)
            labels_flat : (B * num_cat,)

        `ignore_index=-1` silently skips absent-category positions.
        `reduction='sum'` + manual division avoids the NaN that `reduction='mean'`
        produces when ALL entries are -1 (empty batch for every category).
        """
        import torch.nn.functional as F
        B = logits.size(0)
        logits_flat = logits.view(B * self.config.num_categories,
                                  self.config.num_polarities)   # (B*C, P)
        labels_flat = labels.view(B * self.config.num_categories)  # (B*C,)

        valid_mask = labels_flat != -1
        if not valid_mask.any():
            # Entire batch has no category labels (edge-case in tiny smoke tests)
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits_flat, labels_flat,
                               ignore_index=-1, reduction="sum")
        return loss / valid_mask.sum().float()   # mean over valid entries only


    # ──────────────────────────────────────────────────────────────────────
    # Inference helper
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, text: str, tokenizer, device: str = "cpu") -> Dict[str, Any]:
        """
        Run inference on a single sentence.

        Returns:
            {
              "aspect_terms":      [{"term": str, "polarity": str}],
              "aspect_categories": [{"category": str, "polarity": str, "confidence": float}],
            }
        """
        self.eval()
        encoding = tokenizer(
            text,
            max_length        = self.config.max_len,
            padding           = "max_length",
            truncation        = True,
            return_offsets_mapping=True,
            return_tensors    = "pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

        out = self.forward(input_ids, attention_mask, token_type_ids)

        ner_preds   = out["ner_logits"][0].argmax(-1).cpu().tolist()   # (L,)
        cat_probs   = torch.softmax(out["cat_logits"][0], dim=-1).cpu()# (num_cat, num_pol)

        # ── Decode NER spans back to text ─────────────────────────────────
        offsets = encoding["offset_mapping"][0].tolist()
        aspect_terms = _decode_ner_spans(text, ner_preds, offsets, self.config)

        # ── Decode category sentiment predictions ─────────────────────────
        aspect_categories = []
        for c_idx, probs in enumerate(cat_probs):
            best_pol = probs.argmax().item()
            confidence = probs[best_pol].item()
            if confidence > 0.4:                # filter low-confidence predictions
                aspect_categories.append({
                    "category":   self.config.id2cat[c_idx],
                    "polarity":   self.config.id2pol[best_pol],
                    "confidence": round(confidence, 4),
                })

        return {
            "aspect_terms":      aspect_terms,
            "aspect_categories": aspect_categories,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────
    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        # Save BERT weights
        self.bert.save_pretrained(os.path.join(save_dir, "bert"))
        # Save head weights
        torch.save(
            {
                "ner_head":    self.ner_head.state_dict(),
                "cat_head":    self.cat_head.state_dict(),
            },
            os.path.join(save_dir, "heads.pt"),
        )
        logger.info("Model saved to %s", save_dir)

    @classmethod
    def from_pretrained(cls, save_dir: str, config) -> "ABSAModel":
        model = cls(config)
        model.bert = BertModel.from_pretrained(os.path.join(save_dir, "bert"))
        heads = torch.load(
            os.path.join(save_dir, "heads.pt"),
            map_location="cpu",
        )
        model.ner_head.load_state_dict(heads["ner_head"])
        model.cat_head.load_state_dict(heads["cat_head"])
        logger.info("Model loaded from %s", save_dir)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Decode NER spans
# ─────────────────────────────────────────────────────────────────────────────

def _decode_ner_spans(
    text: str,
    ner_preds: List[int],
    offsets: List[Tuple[int, int]],
    config,
) -> List[Dict[str, Any]]:
    """
    Convert B/I/O token labels + offset mapping back to character-level spans.
    Groups consecutive B-ASP / I-ASP tokens into a single span.
    """
    spans: List[Dict[str, Any]] = []
    current_start: Optional[int] = None
    current_end:   Optional[int] = None

    for tok_idx, label_id in enumerate(ner_preds):
        if tok_idx >= len(offsets):
            break
        char_start, char_end = offsets[tok_idx]
        if char_start == 0 and char_end == 0:   # special token
            if current_start is not None:
                _flush_span(text, current_start, current_end, spans)
                current_start = current_end = None
            continue

        tag = config.id2ner.get(label_id, "O")

        if tag == "B-ASP":
            if current_start is not None:
                _flush_span(text, current_start, current_end, spans)
            current_start = char_start
            current_end   = char_end

        elif tag == "I-ASP" and current_start is not None:
            current_end = char_end

        else:   # "O" or unknown
            if current_start is not None:
                _flush_span(text, current_start, current_end, spans)
                current_start = current_end = None

    if current_start is not None:
        _flush_span(text, current_start, current_end, spans)

    return spans


def _flush_span(
    text: str,
    start: int,
    end: int,
    spans: List[Dict[str, Any]],
) -> None:
    term = text[start:end].strip()
    if term:
        spans.append({"term": term, "start": start, "end": end, "polarity": "unknown"})


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer loader
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(config) -> BertTokenizerFast:
    """Load tokenizer from disk if saved, else download and cache."""
    tok_dir = config.tokenizer_dir
    if os.path.isdir(tok_dir) and os.listdir(tok_dir):
        tok = BertTokenizerFast.from_pretrained(tok_dir)
        logger.info("Tokenizer loaded from %s", tok_dir)
    else:
        tok = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
        tok.save_pretrained(tok_dir)
        logger.info("Tokenizer downloaded and saved to %s", tok_dir)
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device(config) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)
