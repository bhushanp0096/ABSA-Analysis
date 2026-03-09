"""
evaluation.py — Metrics computation and visualisation charts.

  - NER F1 / precision / recall  (Task A)
  - Per-category accuracy + macro F1 (Task B)
  - Matplotlib: sentiment distribution bar chart, confusion matrix
  - Plotly: training-curve HTML
"""

import os
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend (safe for server environments)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NER Metrics (Task A)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ner_metrics(
    preds:  List[List[int]],
    labels: List[List[int]],
    id2ner: Dict[int, str],
) -> Dict[str, float]:
    """
    Compute token-level F1/precision/recall for aspect-term NER.
    -100 tokens (padding/specials) are excluded.

    Args:
        preds:  list of per-sentence predicted token label IDs
        labels: list of per-sentence gold token label IDs
    Returns:
        dict with keys: precision, recall, f1, per_label_f1
    """
    flat_preds  = []
    flat_labels = []
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            flat_preds.append(p)
            flat_labels.append(l)

    label_ids = sorted(id2ner.keys())
    label_names = [id2ner[i] for i in label_ids]

    report = classification_report(
        flat_labels,
        flat_preds,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    macro = report.get("macro avg", {})
    return {
        "precision":    macro.get("precision", 0.0),
        "recall":       macro.get("recall",    0.0),
        "f1":           macro.get("f1-score",  0.0),
        "per_label_f1": {
            lbl: report.get(lbl, {}).get("f1-score", 0.0)
            for lbl in label_names
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Category-Sentiment Metrics (Task B)
# ─────────────────────────────────────────────────────────────────────────────

def compute_category_metrics(
    preds:     np.ndarray,   # (N, num_categories)
    labels:    np.ndarray,   # (N, num_categories)   -1 = not present
    id2cat:    Dict[int, str],
    id2pol:    Dict[int, str],
) -> Dict[str, Any]:
    """
    Per-category classification metrics (ignores -1 = 'not mentioned').

    Returns:
        {
          "per_category": { cat_name: {accuracy, f1, support} },
          "macro_f1":     float,
          "overall_accuracy": float,
        }
    """
    n_cats = preds.shape[1]
    per_category: Dict[str, Dict] = {}
    all_preds, all_labels = [], []

    for c in range(n_cats):
        mask = labels[:, c] != -1
        if mask.sum() == 0:
            continue
        p = preds[mask, c]
        l = labels[mask, c]

        acc = (p == l).mean().item()
        f1  = f1_score(l, p, average="macro", zero_division=0)
        per_category[id2cat[c]] = {
            "accuracy": round(acc, 4),
            "f1":       round(f1,  4),
            "support":  int(mask.sum()),
        }
        all_preds.extend(p.tolist())
        all_labels.extend(l.tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    return {
        "per_category":     per_category,
        "macro_f1":         round(float(macro_f1),     4),
        "overall_accuracy": round(float(overall_acc),  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sentiment Distribution Chart  (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

_POLARITY_COLORS = {
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral":  "#2196F3",
    "conflict": "#FF9800",
}

def plot_sentiment_distribution(
    records: List[Dict],
    save_path: str,
) -> str:
    """
    Grouped bar chart: for each aspect category, show counts per polarity.
    Saves as PNG and returns the path.
    """
    cat_pol: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        for c in rec["aspect_categories"]:
            cat_pol[c["category"]][c["polarity"]] += 1

    categories = sorted(cat_pol.keys())
    polarities = ["positive", "negative", "neutral", "conflict"]
    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, pol in enumerate(polarities):
        counts = [cat_pol[cat].get(pol, 0) for cat in categories]
        ax.bar(x + i * width, counts, width, label=pol.capitalize(),
               color=_POLARITY_COLORS.get(pol, "gray"), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Aspect Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Sentiment Distribution per Aspect Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Sentiment distribution chart saved to %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Confusion Matrix  (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    preds:      np.ndarray,    # (N,)  flat predictions
    labels:     np.ndarray,    # (N,)  flat gold labels
    label_names: List[str],
    title:      str,
    save_path:  str,
) -> str:
    """Plot and save a normalised confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(label_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(label_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8)

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training Curves  (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> str:
    """
    Interactive Plotly HTML with sub-plots for loss (train/val) and F1 scores.

    history keys expected: train_loss, val_loss, train_f1_ner, val_f1_ner,
                           train_f1_cat, val_f1_cat  (all lists of length = epochs)
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Total Loss", "NER F1 (Task A)",
            "Category F1 (Task B)", "Val Loss Breakdown",
        ],
    )

    def _add(row, col, name, key, color, dash="solid"):
        if key in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history[key], name=name, mode="lines+markers",
                           line=dict(color=color, dash=dash),
                           hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>"),
                row=row, col=col,
            )

    _add(1, 1, "Train Loss",    "train_loss",    "#1f77b4")
    _add(1, 1, "Val Loss",      "val_loss",      "#ff7f0e", dash="dash")
    _add(1, 2, "Train NER F1",  "train_f1_ner",  "#2ca02c")
    _add(1, 2, "Val NER F1",    "val_f1_ner",    "#98df8a", dash="dash")
    _add(2, 1, "Train Cat F1",  "train_f1_cat",  "#9467bd")
    _add(2, 1, "Val Cat F1",    "val_f1_cat",    "#c5b0d5", dash="dash")
    _add(2, 2, "Val NER Loss",  "val_ner_loss",  "#17becf")
    _add(2, 2, "Val Cat Loss",  "val_cat_loss",  "#e377c2", dash="dash")

    fig.update_layout(
        title_text="ABSA Training Progress",
        title_font_size=18,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        height=680,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Epoch")

    fig.write_html(save_path)
    logger.info("Training curves saved to %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Polarity Distribution Pie  (Matplotlib, quick overview)
# ─────────────────────────────────────────────────────────────────────────────

def plot_polarity_pie(
    records: List[Dict],
    save_path: str,
) -> str:
    """Simple pie chart showing overall polarity distribution."""
    counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        for c in rec["aspect_categories"]:
            counts[c["polarity"]] += 1

    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = [_POLARITY_COLORS.get(l, "gray") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Overall Polarity Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Polarity pie chart saved to %s", save_path)
    return save_path
