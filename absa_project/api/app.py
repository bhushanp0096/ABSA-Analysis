"""
app.py — FastAPI main application.

Endpoints:
  GET  /health            — liveness check
  POST /predict           — single text ABSA prediction
  POST /predict/batch     — batch prediction (up to 32 texts)
  GET  /stats             — dataset statistics
  GET  /docs              — Swagger UI (automatic)
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make sure 'absa_project' is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_parser import parse_xml, compute_dataset_stats
from src.model import ABSAModel, load_tokenizer, get_device
from api.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    AspectTerm,
    AspectCategory,
    StatsResponse,
    HealthResponse,
    ErrorResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Application state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    config:    Config
    model:     Optional[ABSAModel]
    tokenizer: Any
    device:    torch.device
    stats:     Dict

    def __init__(self):
        self.model = None


app_state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load model & compute stats once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model/tokenizer; Shutdown: release resources."""
    cfg = Config()
    app_state.config = cfg
    app_state.device = get_device(cfg)

    # Load tokenizer
    logger.info("Loading tokenizer …")
    app_state.tokenizer = load_tokenizer(cfg)

    # Load model (from saved weights if available, else from pretrained for inference skeleton)
    model_heads = os.path.join(cfg.model_dir, "heads.pt")
    bert_dir    = os.path.join(cfg.model_dir, "bert")
    if os.path.isfile(model_heads) and os.path.isdir(bert_dir):
        logger.info("Loading fine-tuned model from %s …", cfg.model_dir)
        app_state.model = ABSAModel.from_pretrained(cfg.model_dir, cfg)
    else:
        logger.warning(
            "No saved model found at %s — loading untrained BERT backbone. "
            "Run train.py first for meaningful predictions.",
            cfg.model_dir,
        )
        app_state.model = ABSAModel(cfg)

    app_state.model.to(app_state.device)
    app_state.model.eval()

    # Pre-compute dataset statistics
    logger.info("Computing dataset statistics …")
    if os.path.isfile(cfg.xml_file):
        records         = parse_xml(cfg.xml_file)
        app_state.stats = compute_dataset_stats(records)
    else:
        app_state.stats = {"error": "XML data file not found."}
    logger.info("API ready — device: %s", app_state.device)

    yield

    # Shutdown
    logger.info("Shutting down …")
    del app_state.model


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ABSA — Aspect-Based Sentiment Analysis API",
    description = (
        "REST API for performing Aspect-Based Sentiment Analysis on restaurant reviews. "
        "Extracts aspect terms and classifies aspect category sentiment using a fine-tuned BERT model."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Exception handler
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "code": "INTERNAL_ERROR"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper — convert model output to response schema
# ─────────────────────────────────────────────────────────────────────────────

def _build_response(text: str, raw: Dict) -> PredictResponse:
    terms = [
        AspectTerm(
            term     = t["term"],
            polarity = t.get("polarity", "unknown"),
            start    = t["start"],
            end      = t["end"],
        )
        for t in raw.get("aspect_terms", [])
    ]
    cats = [
        AspectCategory(
            category   = c["category"],
            polarity   = c["polarity"],
            confidence = c["confidence"],
        )
        for c in raw.get("aspect_categories", [])
    ]
    return PredictResponse(text=text, aspect_terms=terms, aspect_categories=cats)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Monitoring"],
)
async def health_check():
    """Returns service liveness status and whether the model is loaded."""
    return HealthResponse(
        status       = "ok",
        model_loaded = app_state.model is not None,
        device       = str(app_state.device),
        version      = "1.0.0",
    )


@app.post(
    "/predict",
    response_model   = PredictResponse,
    summary          = "Single-sentence ABSA Prediction",
    tags             = ["Inference"],
    responses        = {400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict(request: PredictRequest):
    """
    Run ABSA on a single restaurant review sentence.

    Returns extracted aspect terms and aspect-category sentiment predictions.
    """
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        raw = app_state.model.predict(
            text      = request.text,
            tokenizer = app_state.tokenizer,
            device    = str(app_state.device),
        )
        return _build_response(request.text, raw)
    except Exception as exc:
        logger.exception("Prediction failed for text: %r", request.text)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.post(
    "/predict/batch",
    response_model   = BatchPredictResponse,
    summary          = "Batch ABSA Prediction",
    tags             = ["Inference"],
    responses        = {400: {"model": ErrorResponse}},
)
async def predict_batch(request: BatchPredictRequest):
    """
    Run ABSA on a list of restaurant review sentences (up to 32).

    Each sentence is processed independently.
    """
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for text in request.texts:
        try:
            raw = app_state.model.predict(
                text      = text,
                tokenizer = app_state.tokenizer,
                device    = str(app_state.device),
            )
            results.append(_build_response(text, raw))
        except Exception as exc:
            logger.error("Batch prediction error for text %r: %s", text, exc)
            # Return empty prediction rather than failing the whole batch
            results.append(PredictResponse(text=text, aspect_terms=[], aspect_categories=[]))

    return BatchPredictResponse(results=results)


@app.get(
    "/stats",
    response_model = StatsResponse,
    summary        = "Dataset Statistics",
    tags           = ["Data"],
)
async def dataset_stats():
    """
    Returns sentiment distribution statistics derived from the training dataset
    (Restaurants_Train_v2.xml).
    """
    if "error" in app_state.stats:
        raise HTTPException(status_code=404, detail=app_state.stats["error"])
    return StatsResponse(**app_state.stats)
