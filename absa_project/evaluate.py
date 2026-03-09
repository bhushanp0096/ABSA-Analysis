"""
evaluate.py — Load a trained ABSA model and run full evaluation on the validation set.

Usage:
    conda run -n agentic_env python evaluate.py
    conda run -n agentic_env python evaluate.py --subset 200
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.config      import Config
from src.data_parser import parse_xml, get_train_val_split, ABSADataset
from src.model       import ABSAModel, load_tokenizer, get_device
from src.evaluation  import (
    compute_ner_metrics,
    compute_category_metrics,
    plot_confusion_matrix,
    plot_sentiment_distribution,
    plot_polarity_pie,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained ABSA model.")
    parser.add_argument("--subset", type=int, default=None,
                        help="Evaluate on the first N val examples (default: all).")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, device, config):
    model.eval()
    all_ner_preds, all_ner_labels = [], []
    all_cat_preds, all_cat_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        ner_labels     = batch["ner_labels"].to(device)
        cat_labels     = batch["cat_labels"].to(device)

        out = model(input_ids, attention_mask, token_type_ids, ner_labels, cat_labels)

        all_ner_preds.extend(out["ner_logits"].argmax(-1).cpu().tolist())
        all_ner_labels.extend(ner_labels.cpu().tolist())
        all_cat_preds.append(out["cat_logits"].argmax(-1).cpu().numpy())
        all_cat_labels.append(cat_labels.cpu().numpy())

    cat_preds_arr  = np.concatenate(all_cat_preds,  axis=0)
    cat_labels_arr = np.concatenate(all_cat_labels, axis=0)

    ner_metrics = compute_ner_metrics(all_ner_preds, all_ner_labels, config.id2ner)
    cat_metrics = compute_category_metrics(cat_preds_arr, cat_labels_arr, config.id2cat, config.id2pol)

    return ner_metrics, cat_metrics, cat_preds_arr, cat_labels_arr


def main():
    args   = parse_args()
    config = Config()
    device = get_device(config)

    # ── Load model ────────────────────────────────────────────────────────
    model_heads = os.path.join(config.model_dir, "heads.pt")
    bert_dir    = os.path.join(config.model_dir, "bert")
    if not (os.path.isfile(model_heads) and os.path.isdir(bert_dir)):
        logger.error("No trained model found at %s. Run train.py first.", config.model_dir)
        sys.exit(1)

    logger.info("Loading model from %s …", config.model_dir)
    model     = ABSAModel.from_pretrained(config.model_dir, config).to(device)
    tokenizer = load_tokenizer(config)

    # ── Load val set ──────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    records = parse_xml(config.xml_file)
    _, val_records = get_train_val_split(records, config.val_ratio, config.seed)

    if args.subset:
        val_records = val_records[:args.subset]
        logger.info("Evaluating on subset: %d examples", len(val_records))

    val_ds     = ABSADataset(val_records, tokenizer, config)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # ── Run evaluation ────────────────────────────────────────────────────
    logger.info("Running evaluation …")
    ner_metrics, cat_metrics, cat_preds, cat_labels = run_evaluation(
        model, val_loader, device, config
    )

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  TASK A — Aspect Term Extraction (NER)")
    print("═" * 60)
    print(f"  Precision : {ner_metrics['precision']:.4f}")
    print(f"  Recall    : {ner_metrics['recall']:.4f}")
    print(f"  F1        : {ner_metrics['f1']:.4f}")
    print("  Per-label F1:")
    for lbl, f1 in ner_metrics["per_label_f1"].items():
        print(f"    {lbl:<10} {f1:.4f}")

    print("\n" + "═" * 60)
    print("  TASK B — Aspect Category Sentiment Classification")
    print("═" * 60)
    print(f"  Macro F1          : {cat_metrics['macro_f1']:.4f}")
    print(f"  Overall Accuracy  : {cat_metrics['overall_accuracy']:.4f}")
    print("  Per-category:")
    for cat, m in cat_metrics["per_category"].items():
        print(f"    {cat:<30} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  n={m['support']}")
    print("═" * 60 + "\n")

    # ── Generate charts ───────────────────────────────────────────────────
    logger.info("Generating evaluation charts …")

    # Confusion matrix for polarity (flatten all categories, mask -1)
    flat_preds  = []
    flat_labels = []
    for c in range(cat_preds.shape[1]):
        mask = cat_labels[:, c] != -1
        flat_preds.extend(cat_preds[mask, c].tolist())
        flat_labels.extend(cat_labels[mask, c].tolist())

    plot_confusion_matrix(
        preds       = np.array(flat_preds),
        labels      = np.array(flat_labels),
        label_names = config.polarities,
        title       = "Polarity Confusion Matrix (Task B)",
        save_path   = os.path.join(config.charts_dir, "confusion_matrix.png"),
    )

    plot_sentiment_distribution(
        records,
        os.path.join(config.charts_dir, "sentiment_distribution.png"),
    )
    plot_polarity_pie(
        records,
        os.path.join(config.charts_dir, "polarity_pie.png"),
    )

    logger.info("Charts saved to %s", config.charts_dir)


if __name__ == "__main__":
    main()
