"""
data_parser.py — XML parsing, text preprocessing, tokenization, and dataset splits.

Handles both Task A (aspect-term NER) and Task B (category-sentiment classification).
"""

import xml.etree.ElementTree as ET
import random
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  XML Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    Parse the SemEval Restaurants XML file.

    Returns a list of dicts:
    {
        "id":               str,
        "text":             str,
        "aspect_terms":     [{"term": str, "polarity": str, "start": int, "end": int}],
        "aspect_categories":[{"category": str, "polarity": str}],
    }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records: List[Dict[str, Any]] = []
    for sentence in root.findall(".//sentence"):
        sid   = sentence.get("id", "")
        text_el = sentence.find("text")
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()

        # --- Aspect Terms (Task A) ---
        # CHANGE 1: broken spans (start >= end) are DROPPED entirely so the model
        # is never taught 'O' on tokens that belong to an unreadable annotation.
        terms: List[Dict] = []
        n_broken = 0
        terms_el = sentence.find("aspectTerms")
        if terms_el is not None:
            for t in terms_el.findall("aspectTerm"):
                term = t.get("term", "").strip()
                polarity = t.get("polarity", "neutral")
                try:
                    start = int(t.get("from", 0))
                    end   = int(t.get("to",   0))
                except ValueError:
                    n_broken += 1
                    continue          # drop: can't produce valid BIO labels
                if end <= start:      # zero-length or inverted span
                    n_broken += 1
                    continue
                if term:
                    terms.append({"term": term, "polarity": polarity,
                                  "start": start, "end": end})
        if n_broken:
            logger.debug("Sentence %s: dropped %d broken span(s)", sid, n_broken)


        # --- Aspect Categories (Task B) ---
        cats: List[Dict] = []
        cats_el = sentence.find("aspectCategories")
        if cats_el is not None:
            for c in cats_el.findall("aspectCategory"):
                category = c.get("category", "").strip()
                polarity = c.get("polarity", "neutral")
                if category:
                    cats.append({"category": category, "polarity": polarity})

        # CHANGE 2: sentences with no aspect terms are kept as EXPLICIT negative
        # samples (all O labels). This is intentional — they teach the NER head
        # what NOT to extract. We log a count so it's visible in training output.
        records.append({
            "id":                sid,
            "text":              text,
            "aspect_terms":      terms,       # may be [] → negative sample
            "aspect_categories": cats,
        })

    n_implicit = sum(1 for r in records if not r["aspect_terms"])
    logger.info(
        "Parsed %d sentences from %s (%d implicit negative-sample rows with no aspect terms)",
        len(records), xml_path, n_implicit,
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Train / Validation Split
# ─────────────────────────────────────────────────────────────────────────────

def get_train_val_split(
    data: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split on dominant aspect category.  Falls back to random split
    if a category has too few examples.
    """
    random.seed(seed)

    # Group by dominant category
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in data:
        if item["aspect_categories"]:
            dominant = item["aspect_categories"][0]["category"]
        else:
            dominant = "none"
        groups[dominant].append(item)

    train, val = [], []
    for grp in groups.values():
        random.shuffle(grp)
        n_val = max(1, int(len(grp) * val_ratio))
        val.extend(grp[:n_val])
        train.extend(grp[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    logger.info("Split → train: %d  val: %d", len(train), len(val))
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Character-offset → token-level NER labels (Task A)
# ─────────────────────────────────────────────────────────────────────────────

def _char_to_token_labels(
    text: str,
    aspect_terms: List[Dict],
    encoding,           # transformers BatchEncoding for a single example
    ner2id: Dict[str, int],
    max_len: int,
) -> List[int]:
    """
    Convert character-level aspect-term spans to wordpiece-level BIO labels.
    -100 is used for special tokens (CLS, SEP, padding) — ignored by CrossEntropyLoss.
    """
    labels = [-100] * max_len

    # Build a char → token_index mapping from the encoding
    char_to_tok: Dict[int, int] = {}
    for tok_idx in range(len(encoding.input_ids)):
        char_span = encoding.token_to_chars(tok_idx)
        if char_span is None:
            continue
        for ci in range(char_span.start, char_span.end):
            char_to_tok[ci] = tok_idx

    # Initialise non-special tokens as "O"
    for tok_idx in range(max_len):
        char_span = encoding.token_to_chars(tok_idx)
        if char_span is not None:
            labels[tok_idx] = ner2id["O"]

    # Annotate aspect term spans
    for term in aspect_terms:
        start_char, end_char = term["start"], term["end"]
        first = True
        for ci in range(start_char, end_char):
            tok_idx = char_to_tok.get(ci)
            if tok_idx is None or tok_idx >= max_len:
                continue
            if labels[tok_idx] == ner2id["O"]:   # don't double-label
                labels[tok_idx] = ner2id["B-ASP"] if first else ner2id["I-ASP"]
                first = False
            elif labels[tok_idx] == ner2id["B-ASP"]:
                first = False

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Category × Polarity multi-label matrix (Task B)
# ─────────────────────────────────────────────────────────────────────────────

def build_category_label(
    aspect_categories: List[Dict],
    cat2id: Dict[str, int],
    pol2id: Dict[str, int],
    num_categories: int,
    num_polarities: int,
) -> torch.Tensor:
    """
    Returns a float tensor of shape (num_categories,) where each element is
    the polarity index for that category (-1 if category not present).
    We model each category as an independent classification problem.
    """
    label = torch.full((num_categories,), -1, dtype=torch.long)
    for item in aspect_categories:
        cat_id = cat2id.get(item["category"])
        pol_id = pol2id.get(item["polarity"], pol2id["neutral"])
        if cat_id is not None:
            label[cat_id] = pol_id
    return label


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ABSADataset(Dataset):
    """
    Tokenises each sentence and returns:
      - input_ids, attention_mask, token_type_ids  (BERT inputs)
      - ner_labels          (shape: max_len)         — Task A
      - cat_labels          (shape: num_categories)  — Task B
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        config,
    ):
        self.records   = records
        self.tokenizer = tokenizer
        self.config    = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        text = rec["text"]

        encoding = self.tokenizer(
            text,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=False,
            return_tensors=None,         # raw dict
        )

        # Re-encode with offsets for NER label alignment
        encoding_with_offsets = self.tokenizer(
            text,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        ner_labels = _char_to_token_labels(
            text              = text,
            aspect_terms      = rec["aspect_terms"],
            encoding          = encoding_with_offsets,
            ner2id            = self.config.ner2id,
            max_len           = self.config.max_len,
        )

        cat_labels = build_category_label(
            aspect_categories = rec["aspect_categories"],
            cat2id            = self.config.cat2id,
            pol2id            = self.config.pol2id,
            num_categories    = self.config.num_categories,
            num_polarities    = self.config.num_polarities,
        )

        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoding.get("token_type_ids",
                                           [0]*self.config.max_len),  dtype=torch.long),
            "ner_labels":     torch.tensor(ner_labels,                 dtype=torch.long),
            "cat_labels":     cat_labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Dataset statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(records: List[Dict]) -> Dict[str, Any]:
    """Compute category/polarity distribution for API /stats endpoint."""
    cat_pol: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_terms = 0
    for rec in records:
        for t in rec["aspect_terms"]:
            total_terms += 1
        for c in rec["aspect_categories"]:
            cat_pol[c["category"]][c["polarity"]] += 1

    return {
        "total_sentences": len(records),
        "total_aspect_terms": total_terms,
        "category_polarity_distribution": {
            cat: dict(pols) for cat, pols in cat_pol.items()
        },
    }
