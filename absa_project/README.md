# Aspect-Based Sentiment Analysis (ABSA) — Project Documentation

> **Dataset**: SemEval-2014 Task 4 — `Restaurants_Train_v2.xml` (3041 sentences)  
> **Environment**: `agentic_env` (Python 3.10, CUDA 12.7, GTX 1650 4GB)  
> **Model**: Fine-tuned `bert-base-uncased` with dual classification heads

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Structure](#2-project-structure)
3. [Dataset & Task Definition](#3-dataset--task-definition)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Data Pipeline — `src/data_parser.py`](#5-data-pipeline--srcdataparserpy)
6. [Model Design — `src/model.py`](#6-model-design--srcmodelpy)
7. [Training Pipeline — `train.py`](#7-training-pipeline--trainpy)
8. [Evaluation & Visualisation — `src/evaluation.py`](#8-evaluation--visualisation--srcevaluationpy)
9. [API Layer — `api/app.py`](#9-api-layer--apiapppy)
10. [GPU Memory Strategy](#10-gpu-memory-strategy)
11. [Configuration — `src/config.py`](#11-configuration--srcconfigpy)
12. [Containerisation — `Dockerfile`](#12-containerisation--dockerfile)
13. [Running the Project](#13-running-the-project)
14. [Key Design Decisions & Rationale](#14-key-design-decisions--rationale)

---

## 1. Problem Statement

**Aspect-Based Sentiment Analysis (ABSA)** goes beyond document-level sentiment ("this review is positive") and asks:

> *"Which specific aspects of a restaurant are mentioned, and what does the reviewer feel about each one?"*

For example, the sentence:  
`"The food was amazing but the service was really slow."`

Should produce:
- **Aspect Term**: `food` → `positive`
- **Aspect Term**: `service` → `negative`  
- **Aspect Category**: `food` → `positive`
- **Aspect Category**: `service` → `negative`

This is significantly harder than standard sentiment analysis because a single sentence can carry **multiple, conflicting sentiments** across different topics simultaneously.

---

## 2. Project Structure

```
absa_project/
├── data/
│   ├── Restaurants_Train_v2.xml   # Raw SemEval training data
│   └── charts/                    # Auto-generated visualisation output
│       ├── sentiment_distribution.png
│       ├── polarity_pie.png
│       └── training_curves.html
│
├── models/
│   ├── bert/                  # Saved fine-tuned BERT weights
│   ├── tokenizer/             # Saved BertTokenizerFast
│   └── heads.pt               # NER head + Category head weights
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Central config dataclass
│   ├── data_parser.py         # XML parsing + PyTorch Dataset
│   ├── model.py               # Dual-head BERT model class
│   └── evaluation.py          # Metrics + chart generation
│
├── api/
│   ├── __init__.py
│   ├── schemas.py             # Pydantic v2 request/response models
│   └── app.py                 # FastAPI application
│
├── train.py                   # End-to-end training script
├── evaluate.py                # Post-training evaluation script
├── requirements.txt
└── Dockerfile
```

---

## 3. Dataset & Task Definition

### SemEval-2014 Task 4 XML Format

```xml
<sentence id="3121">
    <text>But the staff was so horrible to us.</text>
    <aspectTerms>
        <aspectTerm term="staff" polarity="negative" from="8" to="13"/>
    </aspectTerms>
    <aspectCategories>
        <aspectCategory category="service" polarity="negative"/>
    </aspectCategories>
</sentence>
```

### Two Sub-Tasks

| Task | Name | What it solves |
|------|------|----------------|
| **Task A** | Aspect Term Extraction + Sentiment | Find *which exact words* in the sentence are aspect terms |
| **Task B** | Aspect Category + Sentiment | Classify the *high-level topic* and its sentiment |

### Dataset Statistics (parsed)

| Metric | Value |
|--------|-------|
| Total sentences | 3,041 |
| Aspect-term annotations | 3,693 |
| Aspect categories | 5 (food, service, ambience, price, anecdotes/miscellaneous) |
| Sentiment polarities | 4 (positive, negative, neutral, conflict) |
| Train / Val split | 2,738 / 303 (90/10, stratified) |

### Why Stratified Split?

The `anecdotes/miscellaneous` category is far more frequent than `price` or `ambience`. A random split risks having zero examples of a rare category in the validation set, making F1 scores misleading. **Stratified splitting** groups by dominant category first, then draws proportionally — ensuring every category is represented in both sets.

---

## 4. Architecture Deep Dive

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│  BertTokenizerFast                  │  Wordpiece tokenisation
│  max_len=128, return_offsets=True   │  + character offset mapping (for NER)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  BertModel (bert-base-uncased)      │
│  12 layers, 768 hidden, 109M params │
│                                     │
│  input_ids  →  last_hidden_state    │  (B, 128, 768)  — per-token
│              →  pooler_output       │  (B, 768)        — [CLS] token
└─────────────────────────────────────┘
         │                    │
         ▼                    ▼
  ┌─────────────┐      ┌────────────────────────┐
  │  NER Head   │      │  Category Head         │
  │ Dropout(0.1)│      │  Dropout(0.1)          │
  │ Linear      │      │  Linear                │
  │ 768 → 3     │      │  768 → 5×4 = 20        │
  └─────────────┘      └────────────────────────┘
         │                         │
         ▼                         ▼
  Token logits             Reshape → (B, 5, 4)
  (B, 128, 3)         per-category, per-polarity
         │                         │
         ▼                         ▼
   B / I / O tags        food:  positive/neg/neu/con
  → span extraction       service: ...
                          ambience: ...
```

### Why BERT over alternatives?

| Model | Reason NOT chosen |
|-------|-------------------|
| BiLSTM-CRF | No transfer learning; requires more data to reach good F1 |
| RoBERTa | Larger memory footprint — would OOM the GTX 1650 at batch_size ≥ 8 |
| DistilBERT | Faster but 5% lower accuracy on NER tasks vs BERT-base |
| **bert-base-uncased** ✅ | Best accuracy/VRAM tradeoff for 4GB GPU; widely validated on SemEval |

### Why `bert-base-uncased`?

Restaurant reviews are informal and often have **mixed capitalisation** (`great FOOD`, `AMAZING service`). The `uncased` variant lowercases all input, making the model invariant to capitalisation — which reduces vocabulary size and improves generalisation on informal text.

---

## 5. Data Pipeline — `src/data_parser.py`

### Step 1: XML Parsing (`parse_xml`)

**Why** `xml.etree.ElementTree` over `lxml` or `BeautifulSoup`?  
→ It's in Python stdlib — no extra dependency, fast enough for 1.2MB files, and the XML here is well-formed (no HTML entities to unescape).

Each sentence is parsed into:
```python
{
    "id":               "3121",
    "text":             "But the staff was so horrible to us.",
    "aspect_terms":     [{"term": "staff", "polarity": "negative", "start": 8, "end": 13}],
    "aspect_categories":[{"category": "service", "polarity": "negative"}],
}
```

### Step 2: Character-to-Token Label Alignment (`_char_to_token_labels`)

This is the **most technically complex** part of the data pipeline.

**The problem**: The XML gives us character-level spans (`from=8`, `to=13`). BERT uses WordPiece tokenisation which splits words into subword tokens, breaking the 1-to-1 character relationship:

```
Text:     "But  the  staff  was  ..."
Chars:     0123 4567 89012 3456
Tokens:   [CLS] but  the  staff  was  ... [SEP] [PAD]...
```

**The solution**: Use `return_offsets_mapping=True` from `BertTokenizerFast` (a Rust-backed tokenizer). This returns, for each token, the `(char_start, char_end)` range it covers in the original string. We build a `char → token_index` lookup dict, then walk the character span of each aspect term, assigning:
- First character of a term → `B-ASP` (Begin)
- Subsequent characters → `I-ASP` (Inside)
- Everything else → `O` (Outside)
- Special tokens (CLS, SEP, PAD) → `-100` (ignored by PyTorch's CrossEntropyLoss)

**Why `-100` (not 0)?** PyTorch's `CrossEntropyLoss` has an `ignore_index` parameter. Setting labels to `-100` for padding/special tokens tells the loss function to skip those positions entirely — they should never contribute to gradients.

### Step 3: Category Label Tensor (`build_category_label`)

Returns a `(5,)` tensor where:
- Value = polarity index (0–3) if that category is mentioned
- Value = `-1` if category not present in this sentence

**Why `-1` not a one-hot or multi-hot?**  
Each of the 5 categories is treated as an **independent 4-class classification** problem. Using `-1` as the ignore flag (with `CrossEntropyLoss(ignore_index=-1)`) allows a single linear layer to serve all 5 categories simultaneously — much simpler than 5 separate heads.

### Step 4: `ABSADataset` (PyTorch `Dataset`)

`__getitem__` tokenises on the fly (not pre-cached) because:
1. Dataset fits in RAM (3041 × ~500 bytes ≈ 1.5 MB)  
2. Avoids a separate pre-processing step
3. Compatible with `DataLoader(num_workers=2)` — each worker runs its own tokeniser instance

---

## 6. Model Design — `src/model.py`

### Dual-Head BERT (`ABSAModel`)

```python
class ABSAModel(nn.Module):
    bert       = BertModel(...)        # Shared encoder
    ner_head   = Linear(768, 3)        # Task A: B/I/O per token
    cat_head   = Linear(768, 5 × 4)   # Task B: polarity per category
```

**Why share the encoder?**  
Multi-task learning with a shared backbone is a well-established NLP technique. Both tasks encode the same sentence — sharing BERT means:
1. **Half the VRAM** vs two separate BERT models
2. **Regularisation**: Task A forces the encoder to learn span-level features; Task B forces document-level features. Together they improve both tasks vs training each in isolation.

### Loss Functions

**Task A — NER Loss**:
```python
CrossEntropyLoss(ignore_index=-100)(logits.view(-1, 3), labels.view(-1))
```
Standard token classification loss. Flattening to `(B×L, 3)` is required by PyTorch.

**Task B — Category Loss**:
```python
for each category c:
    if all labels[:, c] == -1: skip   # ← crucial fix
    loss += CrossEntropyLoss(ignore_index=-1)(logits[:, c, :], labels[:, c])
total_loss = sum / n_valid_categories
```

> **Bug encountered & fixed**: `CrossEntropyLoss` returns `nan` when **every** sample in a batch has `ignore_index=-1` for a given category. This happened at small batch sizes (8) because some batches contained no examples of `price` or `ambience`. The fix: explicitly check `(labels[:, c] == -1).all()` and skip that category's loss computation.

**Combined loss**:
```python
loss = ner_loss + cat_loss
```
Equal weighting — both tasks are treated as equally important. This can be tuned by adding a `lambda` hyperparameter between them.

### `predict()` Method

For inference, the method:
1. Tokenises with `return_offsets_mapping=True` (needed for span decoding)
2. Runs a single forward pass (no gradient tracking via `@torch.no_grad()`)
3. Decodes NER tags back to character spans using offset mapping
4. Filters category predictions by confidence threshold (> 0.4) — avoids returning low-confidence noise

**Why 0.4 threshold?**  
After softmax over 4 classes, a random predictor would score 0.25 per class. 0.4 requires the model to be somewhat confident it's not random guessing.

### Save/Load — `save_pretrained` / `from_pretrained`

Saves in two parts:
- `bert/` — via HuggingFace's own `save_pretrained` (handles config.json + pytorch_model.bin)
- `heads.pt` — via `torch.save` (just the linear layer state dicts)

**Why not save everything in one file?**  
HuggingFace's save format is ecosystem-compatible (hub upload, other frameworks). Keeping heads separate means you can swap the BERT backbone without re-saving the heads.

---

## 7. Training Pipeline — `train.py`

### Full Training Flow

```
parse_xml() → EDA charts → stratified split → tokenizer → 
ABSADataset → DataLoader → ABSAModel → AdamW + scheduler → 
epoch loop [train_epoch → eval_epoch → VRAM report → save best] → 
training_curves.html
```

### Optimiser: AdamW

**Why AdamW over Adam?**  
AdamW (Adam with decoupled weight decay) is the standard for fine-tuning BERT. Original Adam applies weight decay incorrectly to the adaptive learning rate, which can hurt regularisation. AdamW fixes this decoupling — it's what the BERT paper recommends.

```python
AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

`lr=2e-5` is the canonical BERT fine-tuning learning rate from Devlin et al. (2019). Too high (e.g., 5e-4) destroys pretrained representations; too low (e.g., 1e-6) underfits in 5 epochs.

### Scheduler: Linear Warmup with Decay

```python
get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = 10% of total steps,
    num_training_steps = total_steps,
)
```

**Why warmup?**  
At the start of fine-tuning, the randomly-initialised task heads produce large gradients. If the full learning rate is applied immediately, these gradients destabilise the pretrained BERT weights irreversibly. Warmup ramps the LR from 0 → 2e-5 over the first 10% of steps, letting the heads stabilise before full learning begins.

**Why linear decay after?**  
Learning rate decay prevents overfitting in later epochs. As the model converges, large updates become harmful — decaying the LR reduces them smoothly.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why?**  
In fp16 training, occasional gradient spikes can overflow the float16 range (max ~65504). Clipping to norm=1.0 prevents any single gradient from completely overriding others — standard practice for transformer fine-tuning.

### fp16 Automatic Mixed Precision (AMP)

```python
scaler = GradScaler("cuda", enabled=True)
with autocast("cuda", enabled=True):
    out = model(...)
scaler.scale(scaled_loss).backward()
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()
```

**Why AMP?**  
BERT's attention and linear layers are compute-bound. Running them in `float16` instead of `float32`:
- **Halves VRAM** for activations (768-dim tensors stored in f16 vs f32)
- **2× faster matrix ops** on Tensor Core GPUs

The `GradScaler` is necessary because f16 underflows to zero for very small gradients. It multiplies the loss by a large scalar before backward, then divides after to restore the true gradient magnitudes.

**Measured result**: VRAM dropped from ~3.8GB (f32) → **2.55 GB** (f16) on GTX 1650.

### Gradient Accumulation

```python
# Every batch:
loss = out["loss"] / grad_accum    # divide to keep gradient scale correct
scaler.scale(loss).backward()

# Every grad_accum batches:
scaler.step(optimizer)
optimizer.zero_grad()
```

**Why?**  
GTX 1650 can only fit `batch_size=8` in VRAM. But small batches produce noisy gradient estimates, which slow convergence. Gradient accumulation over `grad_accum=2` steps simulates an **effective batch size of 16** without exceeding VRAM — you accumulate gradients across 2 micro-batches before updating weights.

**Why divide the loss?**  
Without dividing, the accumulated gradient would be 2× the scale it should be. Dividing by `grad_accum` inside the `autocast` block and tracking `full_loss` separately ensures: (a) correct gradient magnitudes, (b) accurate loss logging.

### Best Model Checkpointing

```python
current_f1 = val_f1_cat + val_f1_ner   # combined metric
if current_f1 > best_val_f1:
    model.save_pretrained(cfg.model_dir)
```

**Why combined F1?**  
Using only NER F1 or only category F1 could allow one task to degrade while the other improves. Summing both ensures the saved checkpoint is the best overall model across both tasks simultaneously.

---

## 8. Evaluation & Visualisation — `src/evaluation.py`

### NER Metrics (`compute_ner_metrics`)

Uses `sklearn.metrics.classification_report` over **token-level** predictions (not span-level) because:
- Span-level exact-match F1 is stricter and requires a separate span-matching algorithm
- Token-level F1 is standard for SemEval ABSA comparisons at this stage

Reports per-label F1 for `O`, `B-ASP`, `I-ASP` separately — useful for diagnosing whether the model is failing to begin spans correctly (B-ASP) vs continuing them (I-ASP).

### Category Metrics (`compute_category_metrics`)

```python
for c in range(5):
    mask = labels[:, c] != -1    # only evaluate sentences that mention category c
    f1 = f1_score(labels[mask, c], preds[mask, c], average="macro")
```

**Why mask?**  
Including `-1` labels in the F1 computation would inflate accuracy (most sentences don't mention `price`, so predicting `-1` would be trivially correct). The mask ensures we only score the model on sentences where that category is actually present.

### Charts

| Chart | Library | Why this library |
|-------|---------|-----------------|
| Sentiment distribution bar chart | **Matplotlib** | Publication-quality static PNGs; no JS runtime needed |
| Polarity pie chart | **Matplotlib** | Same — quick EDA overview |
| Training curves | **Plotly** | Interactive HTML — hover to read exact loss/F1 per epoch; shareable without a server |
| Confusion matrix | **Matplotlib** | Seaborn-style heatmap with normalisation; clearer for multi-class comparison |

**Why `matplotlib.use("Agg")`?**  
The `Agg` backend renders to file without opening a display window. This is essential for headless server environments (Docker containers, SSH sessions without X11 forwarding). Without it, matplotlib tries to connect to a display and crashes.

---

## 9. API Layer — `api/app.py`

### FastAPI

**Why FastAPI over Flask?**

| Feature | Flask | FastAPI |
|---------|-------|---------|
| Async support | ❌ (requires extra setup) | ✅ native `async def` |
| Auto Swagger UI | ❌ manual | ✅ `/docs` auto-generated |
| Request validation | ❌ manual | ✅ via Pydantic |
| Type hints | Optional | Required (used for schema generation) |
| Performance | ~1000 req/s | ~3000 req/s |

### Lifespan Model Loading

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    app_state.model    = ABSAModel.from_pretrained(...)
    app_state.tokenizer = load_tokenizer(...)
    app_state.stats    = compute_dataset_stats(parse_xml(...))
    yield
    # SHUTDOWN
    del app_state.model
```

**Why lifespan (not `@app.on_event("startup")`)?**  
`on_event` is deprecated in FastAPI 0.95+. The lifespan context manager is the modern pattern — it uses Python's `contextlib.asynccontextmanager`, making startup/shutdown logic symmetric and testable.

**Why load model at startup (not per-request)?**  
Loading `bert-base-uncased` (420MB) takes 5–15 seconds. Loading it once at startup means zero latency after warmup. The model is placed in `app_state` (module-level singleton), which all requests share safely since `model.eval()` + `@torch.no_grad()` makes inference thread-safe.

### Pydantic v2 Schemas (`api/schemas.py`)

```python
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=512)
```

**Why Pydantic?**  
FastAPI uses Pydantic models to:
1. **Validate** incoming JSON automatically (returns 422 Unprocessable Entity on bad input — no manual try/except needed)
2. **Serialise** responses to JSON
3. **Generate** the OpenAPI schema that powers the `/docs` Swagger UI

**Why `min_length=3`?** A 1–2 character input cannot contain a meaningful aspect term or sentence — rejecting early avoids a model call that would produce nonsense output.

### Batch Endpoint Design

```python
class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., max_length=32)
```

**Why max 32?**  
Each inference call is sequential (the GTX 1650 is already near capacity on a single BERT forward pass). Limiting to 32 prevents request timeouts and protects the server from memory spikes if someone sends 1000 texts.

**Why not true batch inference?**  
True batching (stacking multiple texts into a single forward pass) would be faster but complex: it requires dynamic padding to the longest text in the batch, and at batch_size > 4 risks OOM on the 4GB GPU. Sequential inference keeps memory predictable.

---

## 10. GPU Memory Strategy

### GTX 1650 Constraints

| Resource | Available | Peak Usage | Headroom |
|----------|-----------|------------|---------|
| VRAM | 4.3 GB | 2.55 GB | **40%** |
| CUDA Compute | 7.5 | — | — |

### All Techniques Applied

```
fp16 AMP              → halves activation memory (768-dim f32 → f16)
batch_size = 8        → fits 8 × 128-token sequences in VRAM
grad_accum = 2        → effective batch = 16 (without extra VRAM)
gradient clipping     → prevents f16 overflow from large gradients
pin_memory = True     → faster CPU→GPU transfer (pinned RAM copies)
num_workers = 2       → parallel data loading without taxing CPU
```

### Why These Numbers Specifically?

`batch_size=8, max_len=128` was chosen by estimating:
```
VRAM ≈ params × 2 bytes (f16) + batch × seq_len × hidden × layers × 4 bytes
     ≈ 110M × 2 + 8 × 128 × 768 × 12 × 4
     ≈ 220MB + 378MB ≈ 600MB activations + 420MB model weights
     ≈ ~2.5GB total (matches measured 2.55GB)
```

This left ~1.7GB headroom for PyTorch CUDA allocator overhead and OS GPU usage.

---

## 11. Configuration — `src/config.py`

**Why a dataclass instead of a YAML/JSON config file?**

```python
@dataclass
class Config:
    batch_size: int = 8
    fp16: bool = True
    ...
    def __post_init__(self):
        self.cat2id = {c: i for i, c in enumerate(self.aspect_categories)}
```

Using a `dataclass` gives:
1. **IDE autocomplete** for all config keys (not possible with dict-based YAML)
2. **Type safety** — `batch_size: int` means passing `"8"` as a string raises an error
3. **`__post_init__`** — derived lookup dicts (label maps) are automatically constructed whenever Config is instantiated, so they're always in sync with the source lists
4. **No file I/O overhead** — no YAML parser needed at startup

---

## 12. Containerisation — `Dockerfile`

```dockerfile
FROM python:3.10-slim AS base     # Stage 1: minimal OS
FROM base AS deps                  # Stage 2: install Python deps
COPY requirements.txt . && pip install -r requirements.txt

FROM deps AS app                   # Stage 3: copy source only
COPY src/ api/ data/ models/ train.py evaluate.py .

EXPOSE 8000
HEALTHCHECK --interval=30s ...    # Docker healthcheck
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why multi-stage build?**  
Without multi-stage:
- `gcc`, build tools, and pip cache bloat the final image by ~500MB
- Every source code change triggers a full re-install of all packages

With multi-stage:
- **Stage 2** (deps) is cached as long as `requirements.txt` doesn't change
- **Stage 3** only copies source — so editing `api/app.py` rebuilds in ~2 seconds instead of 5 minutes

**Why `python:3.10-slim`?**  
The `slim` variant strips documentation and locale files off the full Debian image, reducing base size from ~1GB to ~150MB. Python 3.10 matches `agentic_env` — avoiding version-mismatch surprises.

**Why `uvicorn[standard]` (not bare uvicorn)?**  
The `[standard]` extra installs `uvloop` (C-based event loop, 2× faster than Python's default asyncio) and `httptools` (faster HTTP parser). Zero code changes needed — just a faster runtime.

---

## 13. Running the Project

### Environment Setup

```bash
conda activate agentic_env
pip install -r absa_project/requirements.txt
cd absa_project/
```

### Training

```bash
# Full training (5 epochs, ~25 min/epoch on GTX 1650)
python train.py

# Smoke test (2 min total — verify everything works)
python train.py --smoke_test --epochs 2 --no_save

# Custom run
python train.py --epochs 10 --batch_size 8 --grad_accum 4 --lr 1e-5

# CPU only (no GPU)
python train.py --no_fp16
```

### Evaluation (after training)

```bash
python evaluate.py             # Full validation set
python evaluate.py --subset 200  # Faster, first 200 val examples
```

Charts are saved to `data/charts/`:
- `sentiment_distribution.png` — category × polarity bar chart  
- `polarity_pie.png` — overall positive/negative/neutral/conflict breakdown  
- `training_curves.html` — interactive Plotly (open in browser)  
- `confusion_matrix.png` — normalised polarity confusion matrix

### Start API Server

```bash
# Development (auto-reload on file save)
uvicorn api.app:app --port 8000 --reload

# Production
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note**: Use `--workers 1` only. Multiple workers would each load a BERT model (~420MB), OOM-ing the 4GB GPU.

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was great but service was really slow."}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing pasta!", "The waiter was rude.", "Overpriced but cozy."]}'

# Dataset statistics
curl http://localhost:8000/stats
```

### Docker

```bash
# Build
docker build -t absa-api .

# Run
docker run -p 8000:8000 --gpus all absa-api

# Check health
docker ps  # STATUS column shows health check result
```

---

## 14. Key Design Decisions & Rationale

### Summary Table

| Decision | Choice | Why |
|----------|--------|-----|
| Base model | `bert-base-uncased` | Best accuracy/VRAM ratio for 4GB GPU |
| Multi-task | Shared encoder, 2 heads | Halves VRAM; improves both tasks via shared representations |
| Task A formulation | BIO token tagging | Standard NER approach; character offsets mapped to wordpiece tokens |
| Task B formulation | Independent per-category classification | Simpler than multi-label; each category's sentiment is truly independent |
| Loss combination | `ner_loss + cat_loss` (equal weight) | Both tasks equally important; no tuned lambda |
| Optimiser | AdamW, lr=2e-5 | BERT fine-tuning canonical choice |
| Scheduler | Linear warmup (10%) + decay | Prevents head gradients from destabilising BERT at training start |
| Precision | fp16 AMP | 40% VRAM reduction, 2× speed on GTX 1650 |
| Batch strategy | batch=8, accum=2 | Effective batch=16 without exceeding 4GB VRAM |
| nan loss fix | Skip all-masked categories in `_cat_loss` | `CrossEntropyLoss` returns NaN when all labels are `ignore_index` |
| API framework | FastAPI | Async, auto-validation, auto-docs, significantly faster than Flask |
| Config | Python `dataclass` | Type-safe, IDE-friendly, derived maps auto-built in `__post_init__` |
| Visualisation | Matplotlib (static) + Plotly (interactive) | Static for reports; interactive for training monitoring |
| Container | Multi-stage Dockerfile | Separates deps from source; fast rebuilds on code changes |

---

*Generated for `absa_project/` — SemEval-2014 Task 4 Restaurant ABSA System*
