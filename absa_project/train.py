"""
train.py — Full ABSA training pipeline.

Usage:
    conda run -n agentic_env python train.py [options]

Options:
    --epochs      Number of training epochs (default: 5)
    --batch_size  Batch size (default: 16)
    --lr          Learning rate (default: 2e-5)
    --smoke_test  Use only 100 training examples for a quick sanity check
    --no_save     Skip saving the model after training
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# ── ensure absa_project root is on PYTHONPATH ──────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.config     import Config
from src.data_parser import parse_xml, get_train_val_split, ABSADataset, compute_dataset_stats
from src.model      import ABSAModel, load_tokenizer, get_device
from src.evaluation import (
    compute_ner_metrics,
    compute_category_metrics,
    plot_sentiment_distribution,
    plot_training_curves,
    plot_polarity_pie,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train the ABSA dual-head BERT model.")
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--grad_accum", type=int,   default=2,
                        help="Gradient accumulation steps (default: from config).")
    parser.add_argument("--no_fp16",    action="store_true",
                        help="Disable fp16 AMP (use if you hit numerical issues).")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use 100 train examples and 20 val examples for quick testing.")
    parser.add_argument("--no_save",    action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     ABSAModel,
    loader:    DataLoader,
    optimizer: AdamW,
    scheduler,
    device:    torch.device,
    epoch:     int,
    scaler:    GradScaler,
    grad_accum: int = 1,
    use_fp16:  bool = True,
) -> Dict[str, float]:
    model.train()
    total_loss = ner_loss_sum = cat_loss_sum = 0.0
    steps = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        ner_labels     = batch["ner_labels"].to(device)
        cat_labels     = batch["cat_labels"].to(device)

        with autocast("cuda", enabled=use_fp16 and device.type == "cuda"):
            out        = model(input_ids, attention_mask, token_type_ids, ner_labels, cat_labels)
            full_loss  = out["loss"]               # unscaled — for logging
            scaled_loss = full_loss / grad_accum   # scaled — for backward

        scaler.scale(scaled_loss).backward()

        # Only step every grad_accum batches
        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss   += full_loss.item()          # track unscaled loss
        ner_loss_sum += out["ner_loss"].item()
        cat_loss_sum += out["cat_loss"].item()
        steps        += 1

        if steps % 20 == 0:
            vram_used = (
                torch.cuda.memory_reserved(device) / 1e9
                if device.type == "cuda" else 0.0
            )
            logger.info(
                "  [Epoch %d | step %d] loss=%.4f  ner=%.4f  cat=%.4f  VRAM=%.2fGB",
                epoch, steps,
                total_loss / steps,
                ner_loss_sum / steps,
                cat_loss_sum / steps,
                vram_used,
            )

    return {
        "train_loss":     total_loss   / max(steps, 1),
        "train_ner_loss": ner_loss_sum / max(steps, 1),
        "train_cat_loss": cat_loss_sum / max(steps, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model:   ABSAModel,
    loader:  DataLoader,
    device:  torch.device,
    config:  Config,
) -> Dict[str, float]:
    model.eval()
    total_loss = ner_loss_sum = cat_loss_sum = 0.0
    steps = 0

    all_ner_preds:  List[List[int]] = []
    all_ner_labels: List[List[int]] = []
    all_cat_preds:  List[np.ndarray] = []
    all_cat_labels: List[np.ndarray] = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        ner_labels     = batch["ner_labels"].to(device)
        cat_labels     = batch["cat_labels"].to(device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            out = model(input_ids, attention_mask, token_type_ids, ner_labels, cat_labels)
        total_loss   += out["loss"].item()
        ner_loss_sum += out["ner_loss"].item()
        cat_loss_sum += out["cat_loss"].item()
        steps        += 1

        ner_preds = out["ner_logits"].argmax(-1).cpu().tolist()        # (B, L)
        cat_preds = out["cat_logits"].argmax(-1).cpu().numpy()         # (B, num_cat)

        all_ner_preds.extend(ner_preds)
        all_ner_labels.extend(ner_labels.cpu().tolist())
        all_cat_preds.append(cat_preds)
        all_cat_labels.append(cat_labels.cpu().numpy())

    ner_metrics = compute_ner_metrics(all_ner_preds, all_ner_labels, config.id2ner)
    cat_metrics = compute_category_metrics(
        preds   = np.concatenate(all_cat_preds,  axis=0),
        labels  = np.concatenate(all_cat_labels, axis=0),
        id2cat  = config.id2cat,
        id2pol  = config.id2pol,
    )

    return {
        "val_loss":     total_loss   / max(steps, 1),
        "val_ner_loss": ner_loss_sum / max(steps, 1),
        "val_cat_loss": cat_loss_sum / max(steps, 1),
        "val_f1_ner":   ner_metrics["f1"],
        "val_f1_cat":   cat_metrics["macro_f1"],
        "val_acc_cat":  cat_metrics["overall_accuracy"],
        "per_category": cat_metrics["per_category"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = Config()

    # Override config with CLI args
    if args.epochs:     cfg.epochs     = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.learning_rate = args.lr
    if args.grad_accum: cfg.gradient_accumulation_steps = args.grad_accum
    if args.no_fp16:    cfg.fp16 = False

    device = get_device(cfg)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        vram_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(device), vram_total)
        logger.info("fp16 AMP: %s | grad_accum: %d | effective_batch: %d",
                    cfg.fp16, cfg.gradient_accumulation_steps,
                    cfg.batch_size * cfg.gradient_accumulation_steps)

    # ── 1. Parse dataset ──────────────────────────────────────────────────
    logger.info("Parsing XML dataset …")
    records = parse_xml(cfg.xml_file)

    # Print & save dataset stats before training
    stats = compute_dataset_stats(records)
    logger.info(
        "Dataset: %d sentences | %d aspect-term annotations",
        stats["total_sentences"], stats["total_aspect_terms"],
    )

    # ── 2. Generate EDA charts ────────────────────────────────────────────
    logger.info("Generating EDA charts …")
    plot_sentiment_distribution(
        records,
        os.path.join(cfg.charts_dir, "sentiment_distribution.png"),
    )
    plot_polarity_pie(
        records,
        os.path.join(cfg.charts_dir, "polarity_pie.png"),
    )

    # ── 3. Train/val split ────────────────────────────────────────────────
    train_records, val_records = get_train_val_split(records, cfg.val_ratio, cfg.seed)

    if args.smoke_test:
        logger.warning("SMOKE TEST mode: using 100 train / 20 val samples only.")
        train_records = train_records[:100]
        val_records   = val_records[:20]

    # ── 4. Tokenizer ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer …")
    tokenizer = load_tokenizer(cfg)

    # ── 5. Datasets & DataLoaders ─────────────────────────────────────────
    train_ds = ABSADataset(train_records, tokenizer, cfg)
    val_ds   = ABSADataset(val_records,   tokenizer, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ── 6. Model ──────────────────────────────────────────────────────────
    logger.info("Initialising model …")
    model = ABSAModel(cfg).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameters: %s total | %s trainable",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    # ── 7. Optimiser & scheduler ──────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Adjust total_steps for gradient accumulation
    steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    # AMP scaler — no-op when fp16=False or on CPU
    use_fp16 = cfg.fp16 and device.type == "cuda"
    scaler   = GradScaler("cuda", enabled=use_fp16)
    logger.info("Total optim steps: %d | Warmup: %d | fp16: %s",
                total_steps, warmup_steps, use_fp16)

    # ── 8. Training loop ──────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_ner_loss": [], "train_cat_loss": [],
        "val_loss":   [], "val_ner_loss":   [], "val_cat_loss":   [],
        "val_f1_ner": [], "val_f1_cat":     [],
    }
    best_val_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        logger.info("── Epoch %d / %d ──────────────────────────────────────", epoch, cfg.epochs)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            scaler    = scaler,
            grad_accum = cfg.gradient_accumulation_steps,
            use_fp16  = use_fp16,
        )
        val_metrics = eval_epoch(model, val_loader, device, cfg)

        if device.type == "cuda":
            vram_used = torch.cuda.memory_reserved(device) / 1e9
            logger.info("  VRAM usage after epoch %d: %.2f GB / %.1f GB",
                        epoch, vram_used,
                        torch.cuda.get_device_properties(device).total_memory / 1e9)

        for k, v in {**train_metrics, **val_metrics}.items():
            if k in history:
                history[k].append(v)

        elapsed = time.time() - t0
        logger.info(
            "  Epoch %d done (%.1fs): "
            "train_loss=%.4f | val_loss=%.4f | val_f1_ner=%.4f | val_f1_cat=%.4f",
            epoch, elapsed,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            val_metrics["val_f1_ner"],
            val_metrics["val_f1_cat"],
        )

        # Per-category breakdown
        for cat, m in val_metrics.get("per_category", {}).items():
            logger.info("    %-30s acc=%.3f  f1=%.3f  n=%d", cat, m["accuracy"], m["f1"], m["support"])

        # Save best model
        current_f1 = val_metrics["val_f1_cat"] + val_metrics["val_f1_ner"]
        if not args.no_save and current_f1 > best_val_f1:
            best_val_f1 = current_f1
            logger.info("  ↳ New best model (combined F1 = %.4f). Saving …", best_val_f1)
            model.save_pretrained(cfg.model_dir)
            tokenizer.save_pretrained(cfg.tokenizer_dir)

    # ── 9. Training curves ────────────────────────────────────────────────
    logger.info("Saving training curves …")
    plot_training_curves(
        history,
        os.path.join(cfg.charts_dir, "training_curves.html"),
    )

    logger.info("✓ Training complete. Best combined F1 = %.4f", best_val_f1)
    logger.info("  Charts saved to: %s", cfg.charts_dir)
    logger.info("  Model saved to:  %s", cfg.model_dir)


if __name__ == "__main__":
    main()
