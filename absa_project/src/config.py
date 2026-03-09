"""
config.py — Centralised configuration for ABSA project.
All paths, hyperparameters, and label maps live here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ─── Project root (absa_project/) ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: str = os.path.join(_HERE, "data")
    xml_file: str = os.path.join(_HERE, "data", "Restaurants_Train_v2.xml")
    charts_dir: str = os.path.join(_HERE, "data", "charts")
    model_dir: str = os.path.join(_HERE, "models")
    tokenizer_dir: str = os.path.join(_HERE, "models", "tokenizer")

    # ── Pretrained base ────────────────────────────────────────────────────
    pretrained_model_name: str = "bert-base-uncased"

    # ── Training hyperparameters ───────────────────────────────────────────
    max_len: int = 128
    batch_size: int = 8            # safe for GTX 1650 4GB VRAM
    gradient_accumulation_steps: int = 2   # effective batch = 8 × 2 = 16
    fp16: bool = True              # AMP mixed-precision — halves VRAM usage
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    val_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"          # "auto" | "cpu" | "cuda"

    # ── Aspect categories (Task B) ─────────────────────────────────────────
    aspect_categories: List[str] = field(default_factory=lambda: [
        "food",
        "service",
        "ambience",
        "price",
        "anecdotes/miscellaneous",
    ])

    # ── Sentiment polarities ───────────────────────────────────────────────
    polarities: List[str] = field(default_factory=lambda: [
        "positive",
        "negative",
        "neutral",
        "conflict",
    ])

    # ── NER tags for aspect-term extraction (Task A) ───────────────────────
    # B-ASP = Begin, I-ASP = Inside, O = Outside
    ner_labels: List[str] = field(default_factory=lambda: [
        "O",
        "B-ASP",
        "I-ASP",
    ])

    # ── Derived label maps (built in __post_init__) ───────────────────────
    cat2id: Dict[str, int] = field(init=False)
    id2cat: Dict[int, str] = field(init=False)
    pol2id: Dict[str, int] = field(init=False)
    id2pol: Dict[int, str] = field(init=False)
    ner2id: Dict[str, int] = field(init=False)
    id2ner: Dict[int, str] = field(init=False)

    def __post_init__(self):
        self.cat2id = {c: i for i, c in enumerate(self.aspect_categories)}
        self.id2cat = {i: c for c, i in self.cat2id.items()}
        self.pol2id = {p: i for i, p in enumerate(self.polarities)}
        self.id2pol = {i: p for p, i in self.pol2id.items()}
        self.ner2id = {t: i for i, t in enumerate(self.ner_labels)}
        self.id2ner = {i: t for t, i in self.ner2id.items()}

        # Ensure output dirs exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

    @property
    def num_categories(self) -> int:
        return len(self.aspect_categories)

    @property
    def num_polarities(self) -> int:
        return len(self.polarities)

    @property
    def num_ner_labels(self) -> int:
        return len(self.ner_labels)

    @property
    def num_cat_labels(self) -> int:
        """Total output units for category-sentiment head = cats × polarities."""
        return self.num_categories * self.num_polarities
