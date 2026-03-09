"""
schemas.py — Pydantic request/response models for the ABSA FastAPI service.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Restaurant review sentence to analyse.",
        examples=["The food was great but the service was really slow."],
    )


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of sentences (max 32 per request).",
    )

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v: List[str]) -> List[str]:
        for t in v:
            if not t.strip():
                raise ValueError("Each text must be non-empty.")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────────────────

class AspectTerm(BaseModel):
    term:     str   = Field(..., description="Extracted aspect term text.")
    polarity: str   = Field(..., description="Predicted polarity (unknown when NER-only).")
    start:    int   = Field(..., description="Start character offset in original text.")
    end:      int   = Field(..., description="End character offset in original text.")


class AspectCategory(BaseModel):
    category:   str   = Field(..., description="Aspect category (food/service/ambience/price/anecdotes).")
    polarity:   str   = Field(..., description="Predicted polarity for this category.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")


class PredictResponse(BaseModel):
    text:              str                    = Field(..., description="Original input text.")
    aspect_terms:      List[AspectTerm]       = Field(default_factory=list)
    aspect_categories: List[AspectCategory]   = Field(default_factory=list)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# ─────────────────────────────────────────────────────────────────────────────
# Stats / Health Models
# ─────────────────────────────────────────────────────────────────────────────

class CategoryStats(BaseModel):
    positive: int = 0
    negative: int = 0
    neutral:  int = 0
    conflict: int = 0


class StatsResponse(BaseModel):
    total_sentences:                  int
    total_aspect_terms:               int
    category_polarity_distribution:   Dict[str, Dict[str, int]]


class HealthResponse(BaseModel):
    status:       str          = "ok"
    model_loaded: bool
    device:       str
    version:      str          = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    code:   Optional[str] = None
