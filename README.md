<div align="center">
  <img src="client/assets/logo.webp" alt="Logo" width="150"/>

  # Unified AI Platform

  **An end-to-end, self-hosted ML system for Synthetic Data Generation, Computer Vision, LLM Fine-Tuning, and Tabular AutoML — all behind a single Vanilla JS frontend.**

  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
  [![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-blue?style=for-the-badge)](https://ultralytics.com/)
  [![Unsloth](https://img.shields.io/badge/Unsloth-LLaMA%20QLora-purple?style=for-the-badge)](https://github.com/unslothai/unsloth)
  [![AutoGluon](https://img.shields.io/badge/AutoGluon-Tabular%20AutoML-green?style=for-the-badge)](https://auto.gluon.ai/)
  [![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://unified-model-interface.vercel.app/)

  > **The frontend is deployed at [https://unified-model-interface.vercel.app/](https://unified-model-interface.vercel.app/)**
  > The Vercel deployment only hosts the static UI. You must run the FastAPI backend locally and point the frontend to your machine for full functionality.
</div>

---

## What This Project Is

The Unified AI Platform is a fully integrated ML engineering toolkit. Its purpose is to remove the friction from every stage of the machine learning lifecycle:

- **No data?** Use the Synthetic Data Factory to generate tens of thousands of realistic, statistically correlated rows instantly.
- **Have raw data?** The ingestion and understanding pipeline automatically classifies your data, determines the right ML task type, and routes it to the correct training adapter.
- **Want to train a model?** Choose from Tabular (AutoGluon), LLM fine-tuning (Unsloth/QLora), or Computer Vision (YOLOv8) — each wrapped behind a unified adapter interface.
- **Have documents?** Build a RAG (Retrieval-Augmented Generation) pipeline over your PDFs and query them via natural language.
- **Built a vision model?** Get instant predictions and inference directly in the browser.

All of this is controlled from a single, zero-dependency Vanilla JS frontend — no React, no npm, just HTML + JS talking directly to FastAPI via REST.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Vanilla JS / CSS)                        │
│           Static deployment: unified-model-interface.vercel.app       │
│   OR run locally: cd client && python3 -m http.server 3000           │
└───────────────────────┬───────────────────────────────────────────────┘
                        │ REST / JSON (HTTP)
                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│               FASTAPI APPLICATION  (backend/main.py)                  │
│  uvicorn backend.main:app  —  port 8000                               │
│                                                                        │
│  Startup:  init_db() → create storage dirs → probe_gpus()            │
│  CORS:     allow_origins=["*"] — any origin can connect               │
│                                                                        │
│  Routers mounted:                                                      │
│  ├── /api/*          → backend/api.py        (all main endpoints)     │
│  ├── /api/synthetic/* → backend/synthetic/api.py (data factory)      │
│  └── /status         → backend/status.py    (health + GPU status)    │
└────────┬──────────────────────────────────────────────────────────────┘
         │
         ├── Synthetic Data Factory   (/backend/synthetic/)
         ├── Custom Vision Engine     (/backend/custom_vision_engine.py)
         ├── Training Adapters        (/backend/adapters/)
         ├── RAG Engine               (/backend/rag/)
         ├── Orchestrator             (/backend/orchestrator/)
         ├── Dataset Ingestion        (/backend/ingestion/, /backend/dataset/)
         └── Model Registry           (/backend/registry/, /backend/db.py)
```

---

## Deep Technical Breakdown

### 1. Synthetic Data Factory (`backend/synthetic/`)

This is the flagship system. The core problem it solves: traditional LLM data generation is brutally slow because it generates one cell at a time. We bypass this entirely.

**The Pipeline:**

```
User Prompt (natural language)
        │
        ▼
[POST /api/synthetic/schema]
        │
        ▼
GroqProvider.generate_schema()  →  Calls Groq (Llama-3 via LPU)
        │
        ▼  Returns a JSON object with:
        │  ├── "columns": [{name, type, description, generation_strategy: "python"|"llm"}]
        │  ├── "python_code": "def generate_base_data(num_rows): ..."   ← KEY STEP
        │  └── "target_rows": N
        │
[POST /api/synthetic/generate]  →  Spawns background thread
        │
        ▼
background_generation_task() in engine.py:
        │
        ├── exec(python_code)              ← Compile and run LLM-generated Python
        │   │                               This generates N rows in milliseconds
        │   │   Uses: random.gauss(), random.choices(), pandas
        │   │   Real-world constraints: Gaussian distributions, realistic means
        │   │
        │   └── raw_records = generate_base_data(N)
        │
        ├── [IF any column has generation_strategy=="llm"]
        │   └── Chunked LLM Fill (chunks of 10 rows sent to Groq)
        │       ├── GroqProvider.fill_semantic_fields()
        │       ├── API Key Rotation: auto-cycles GROQ_API_KEY_1..N on 429/413
        │       └── Temperature 0.85 + high-variance prompt for text diversity
        │
        ├── pd.DataFrame(records)
        │   └── Type casting: uuid→sequential ID, float→.round(2), bool→True/False
        │
        └── df.to_parquet("/tmp/{job_id}.parquet")

[GET /api/synthetic/status/{job_id}]  →  Polls progress (valid rows count)
[GET /api/synthetic/download/{job_id}/csv|json]  →  Reads parquet, streams response
```

**Key Design Decisions:**
- The LLM writes **deterministic Python math code**, not data. The Python runtime generates the actual data. This makes 10,000 rows as fast as 10 rows.
- Variables like birth weight, height, test scores use `random.gauss(mu, sigma)` with factual real-world means. No `min()`/`max()` clamping which causes artificial data spikes.
- Multiple `GROQ_API_KEY` environment variables form a rotating pool. When rate limited (HTTP 413/429), the system catches the exception and switches to the next key automatically, without any user-facing failure.

**Files:**
| File | Role |
|------|------|
| `synthetic/llm_provider.py` | Groq API client, key pool management, prompt engineering for schema generation and semantic fill |
| `synthetic/engine.py` | Background threading, Python sandbox execution, parquet storage, job state management |
| `synthetic/api.py` | FastAPI router exposing `/schema`, `/generate`, `/status`, `/download` |

---

### 2. Custom Vision Engine (`backend/custom_vision_engine.py`)

A fully in-house PyTorch classifier built for rapid transfer learning on user-supplied image classes. This is completely independent of any third-party AutoML and uses:

- **Backbone selection:** MobileNetV3-Small (default, fast), ResNet-50, EfficientNet-B0 — all pre-trained on ImageNet from `torchvision.models`.
- **Custom classification head:** Replaces the backbone's final linear layer with `nn.Linear(backbone_out_features, num_classes)`.
- **In-memory dataset:** `InMemoryDataset` (a `torch.utils.data.Dataset` subclass) holds PIL images and integer labels in RAM, avoiding disk I/O overhead during training.
- **Training transforms:** `RandomHorizontalFlip` + `ColorJitter` for light augmentation. Eval uses clean `Resize(224,224)` + `Normalize(ImageNet mean/std)`.
- **Device handling:** `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — automatically leverages GPU if present.
- **In-memory model cache:** `_ACTIVE_MODEL` global caches the loaded model to serve sub-10ms inference without reloading weights from disk on every prediction request.
- **Export format:** Model weights persisted as `.pth` files under `storage/models/custom_classifier/{model_id}/`.

```
POST /api/teachable/train  →  Accepts base64 image batches per class label
                              Trains in background thread
                              Returns model_id

POST /api/teachable/predict  →  Loads cached model
                                Runs single forward pass
                                Returns {label, confidence, top_k_predictions}
```

---

### 3. Training Adapter Layer (`backend/adapters/`)

The adapter layer abstracts three completely different ML backends (AutoGluon, Unsloth, Ultralytics) behind a single `BackendAdapter` interface defined in `adapters/base.py`. Every adapter implements:

```python
class BackendAdapter:
    def capabilities(self) -> dict          # What tasks/models/formats are supported
    def estimate_resources(self, ...) -> ResourceEstimate  # VRAM/RAM/disk estimate
    def prepare(self, dataset_path, config) -> Path        # Dataset preprocessing
    def train(self, dataset_path, config) -> TrainingResult  # Core training
    def evaluate(self, ...) -> EvaluationResult            # Metrics
    def export(self, ...)                                  # Model export
```

**AutoGluon Adapter (`adapters/autogluon.py`) — Tabular AutoML**
- Wraps `autogluon.tabular.TabularPredictor`.
- `prepare()`: Reads CSV/parquet, does an 80/20 train-val split via `sklearn.model_selection.train_test_split`, saves to `storage/processed/`.
- `train()`: Calls `TabularPredictor.fit()` which internally runs an ensemble of LightGBM, XGBoost, CatBoost, Random Forests, Neural Networks, and picks the best combination automatically.
- Supports: classification, regression, ranking. Target column is auto-detected or passed from the orchestrator.
- Inference: `/api/model/predict` loads the predictor, accepts a dict of feature values, returns a float prediction. Model is cached in `_active_tabular_model` global.

**Unsloth Adapter (`adapters/unsloth.py`) — LLM Fine-Tuning**
- Wraps `unsloth` library for memory-efficient LLM fine-tuning via LoRA/QLoRA.
- **Resource estimation arithmetic:** Extracts parameter count from model name (e.g. `7b` → 7.0B params), calculates VRAM as `params_b * 1024 * bytes_per_param`. For 4-bit QLoRA: `0.55 bytes/param`. For bf16: `2.0 bytes/param`. Adds optimizer overhead + activation buffer.
- **Supported models:** LLaMA-3.2 (1B, 3B), LLaMA-3.1 (8B, 70B), Qwen2.5 (0.5B–72B), Mistral-7B, Gemma-2, Phi-3.5 — all as BnB-4bit quantized Unsloth checkpoints.
- **Supported export formats:** `lora_dir` (adapter weights only), `gguf` (quantized for llama.cpp), `merged_16bit`, `merged_4bit`.
- Includes a Python 3.14 compatibility monkeypatch for the `dill` pickler used by HuggingFace datasets.

**Ultralytics Adapter (`adapters/ultralytics.py`) — Computer Vision Training**
- Wraps `ultralytics.YOLO` for training custom object detectors and image classifiers.
- Supported models: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8n-cls`, `yolov8s-cls`.
- Accepts user-uploaded YOLO-format zip archives (images + labels in YOLO annotation format).
- VRAM estimates: nano=512MB, small=1GB, medium=2GB, large=4GB.
- Export formats: `.pt` (PyTorch), ONNX, TorchScript.

---

### 4. Orchestrator & Problem Formulator (`backend/orchestrator/`)

The orchestrator is the "brain" that routes a user's uploaded data and goal statement to the correct adapter without the user needing to know what adapter exists.

**`problem_formulator.py` Flow:**
1. Reads the `understanding_report` (produced by `understanding/engine.py`) containing file type counts, column names, data statistics.
2. Calls Groq to generate an initial `task_spec` JSON with `task_type`, `modality`, `target_column`.
3. **Runs deterministic heuristic overrides** on top of the LLM output to prevent hallucinations:
   - PDF + Q&A keywords → force `task_type = "rag"`
   - Chat logs + style keywords → force `task_type = "fine_tuning"`
   - CSV/parquet present → force `modality = "tabular"`, default `task_type = "classification"`
   - Image files or vision keywords → force `modality = "image"`, default `task_type = "object_detection"`
4. Returns a normalized spec: `{needs_training, task_type, modality, target_column, rationale}`.

**`validation_gate.py`:** Before dispatching a training job, checks available VRAM via `gpu_probe.py` against the adapter's `estimate_resources()`. Rejects jobs that would OOM the GPU.

**`groq_client.py` + `tools.py`:** Function-calling interface for LLM-based tool use within orchestration flows (e.g. asking Groq to fill in missing schema fields).

---

### 5. RAG Engine (`backend/rag/`)

A from-scratch RAG implementation using FAISS for vector search and `sentence-transformers` for embeddings. No LangChain, no LlamaIndex.

**Chunking (`rag/chunking.py`):**
- Word-based sliding window. Default: chunk_size=500 words, overlap=50 words.
- Each chunk carries a copy of the document's metadata dict for source attribution.

**Embeddings (`rag/embeddings.py`):**
- Uses `sentence-transformers` model `all-MiniLM-L6-v2` which produces 384-dimensional vectors.

**Vector Store (`rag/vector_store.py`):**
- Wraps a `faiss.IndexFlatL2` (L2 distance, exact nearest-neighbor, no approximation).
- `add()`: Accepts list of float embeddings and corresponding chunks. Converts to `np.float32` and calls `faiss.index.add()`.
- `retrieve(query_embedding, k=5)`: Runs `index.search()`, returns top-k chunk dicts.
- `save()/load()`: Persists `index.faiss` binary and `chunks.json` to disk for resumable RAG sessions.

**Generator (`rag/generator.py`):**
- Takes retrieved chunks, formats them into a context window, and calls the LLM (Groq) with the user's question + context injected into the system prompt.

---

### 6. Dataset Ingestion & Chat Parsing (`backend/ingestion/`, `backend/dataset/`)

**`ingestion/engine.py`:** Handles multi-format file ingestion. Normalizes CSV, JSON, Parquet, XLSX, and PDF uploads into a standard internal representation stored in `storage/raw/`.

**`dataset/builder.py`:** Constructs the final dataset artifact from ingested raw files. Handles deduplication, schema validation, and outputs a standardized parquet to `storage/processed/`.

**`dataset/chat_parser.py` — ChatGPT Export Parser:**
A specialized parser that converts ChatGPT's `conversations.json` export format into `ShareGPT` JSONL format required by Unsloth/LLaMA-Factory for fine-tuning on personal conversation style.
- Traverses the ChatGPT conversation graph backwards from the `current_node` leaf to the root, reconstructing the linear conversation path.
- Maps OpenAI roles (`user`, `assistant`, `tool`) → ShareGPT roles (`human`, `gpt`, `tool`).
- Deduplicates conversations by content hash.
- Outputs clean JSONL with `conversations: [{from, value}]` format.

---

### 7. GPU Probe (`backend/gpu_probe.py`)

Called at startup and before every training job dispatch. Shells out to `nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv` and parses the output into typed `GPUInfo` dataclasses. Returns an empty list gracefully if `nvidia-smi` is not found (CPU-only environments). The `get_max_free_vram_mb()` helper is used by adapters and the validation gate to gate training jobs by available VRAM.

---

### 8. Model Registry & Storage (`backend/registry/`, `backend/db.py`, `backend/storage/`)

**`db.py`:** SQLAlchemy ORM models (`TrainingRun`, `ModelArtifact`) backed by SQLite at `storage/db/unified.sqlite3`. Records experiment metadata: `experiment_id`, hyperparameters JSON, status, start/end timestamps, VRAM used.

**`registry/capabilities.yaml`:** Static YAML file declaring every supported model's supported tasks, training methods, VRAM requirements, and sub-model variants. Loaded by `registry/loader.py` and consumed by adapters via `get_model_capabilities()`.

**`storage/` layout:**
```
storage/
├── raw/            ← User-uploaded datasets (original format)
├── processed/      ← Normalized parquet files post-ingestion
├── models/         ← Trained model weights and artifacts
├── experiments/    ← Per-experiment training logs, configs, checkpoints
└── logs/           ← Server-side training logs
```

---

## How to Use

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU recommended for LLM fine-tuning (CPU works for tabular/small vision)
- At least one Groq API key from [console.groq.com](https://console.groq.com/)

### Step 1: Configure Environment

Create `.env` at the root of the project:
```env
# Groq API key pool — system auto-rotates on rate limits
GROQ_API_KEY_1=gsk_your_key_1
GROQ_API_KEY_2=gsk_your_key_2
GROQ_API_KEY_3=gsk_your_key_3
GROQ_API_KEY_4=gsk_your_key_4

# Model to use for all LLM calls (default: Llama 3 on Groq LPU)
GROQ_MODEL=openai/gpt-oss-120b
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start the Backend
```bash
source .env
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be live at `http://localhost:8000`. The `--reload` flag restarts the server on any code change.

On startup, the server will:
1. Initialize the SQLite database and create all tables.
2. Create any missing `storage/` subdirectories.
3. Run `nvidia-smi` to detect available GPUs and log their free VRAM.

### Step 4: Connect the Frontend
**Option A — Use the Vercel Deployment (Recommended):**
Navigate to [https://unified-model-interface.vercel.app/](https://unified-model-interface.vercel.app/).
In the settings panel, set the Backend URL to `http://localhost:8000`. The Vercel frontend will route all API calls to your local backend.

**Option B — Run the Frontend Locally:**
```bash
cd client
python3 -m http.server 3000
```
Navigate to `http://localhost:3000`.

### Step 5: Use the Platform

**Synthetic Data Factory:**
1. Describe your dataset in plain English (e.g., *"1000 rows of employee salary data with age, department, years of experience and salary"*).
2. Select the number of rows.
3. Click **Generate Schema** — the LLM will design the schema and Python generation logic.
4. Review the schema. Click **Generate Dataset**.
5. Watch the real-time progress bar as rows are generated.
6. Download as CSV or JSON.

**Vision Predict:**
1. Upload any image.
2. Choose a pre-trained YOLO model or a custom classifier.
3. Get instant bounding boxes, class labels, and confidence scores.

**Train a Model:**
1. Upload your dataset (CSV for tabular, zip of images for vision, JSONL for LLM).
2. Describe your goal.
3. The orchestrator auto-selects the correct adapter and proposes a training plan.
4. Approve and start training. Monitor progress in real-time.
5. Download the trained model weights when complete.

---

## Project Structure

```
Backend/
├── backend/
│   ├── main.py                 ← FastAPI app entrypoint, router mounting, lifespan
│   ├── api.py                  ← All main API endpoints (training, vision, RAG, etc.)
│   ├── config.py               ← Pydantic Settings — loads .env, declares all paths
│   ├── db.py                   ← SQLAlchemy models (TrainingRun, ModelArtifact)
│   ├── gpu_probe.py            ← nvidia-smi wrapper for VRAM detection
│   ├── custom_vision_engine.py ← In-house PyTorch transfer learning classifier
│   │
│   ├── synthetic/              ← Synthetic Data Factory
│   │   ├── api.py              ← /api/synthetic/* router
│   │   ├── engine.py           ← Background job runner, Python sandbox exec
│   │   └── llm_provider.py     ← Groq client, API key pool, prompt engineering
│   │
│   ├── adapters/               ← Unified ML adapter interface
│   │   ├── base.py             ← BackendAdapter ABC + shared dataclasses
│   │   ├── autogluon.py        ← Tabular AutoML (AutoGluon)
│   │   ├── unsloth.py          ← LLM fine-tuning (LoRA/QLoRA via Unsloth)
│   │   ├── ultralytics.py      ← Computer Vision (YOLOv8)
│   │   ├── rag.py              ← RAG pipeline adapter
│   │   └── llama_factory.py    ← LLaMA-Factory adapter
│   │
│   ├── orchestrator/           ← Routing & validation
│   │   ├── problem_formulator.py ← LLM + heuristics → task spec
│   │   ├── validation_gate.py  ← VRAM gate, pre-training checks
│   │   ├── groq_client.py      ← Groq function-calling client
│   │   └── tools.py            ← Tool definitions for orchestrator LLM
│   │
│   ├── rag/                    ← RAG engine (no LangChain)
│   │   ├── chunking.py         ← Sliding-window document chunker
│   │   ├── embeddings.py       ← sentence-transformers embedding generator
│   │   ├── vector_store.py     ← FAISS L2 index with persist/load
│   │   └── generator.py        ← Context injection + Groq answer generation
│   │
│   ├── dataset/                ← Dataset processing
│   │   ├── builder.py          ← Raw → processed parquet normalizer
│   │   └── chat_parser.py      ← ChatGPT export → ShareGPT JSONL converter
│   │
│   ├── ingestion/              ← File upload handling (CSV/JSON/PDF/images)
│   ├── registry/               ← Model capability declarations (YAML + loader)
│   ├── understanding/          ← Dataset analysis engine
│   └── storage/                ← Local data lake (raw, processed, models, logs)
│
├── client/
│   ├── index.html              ← Single-page application markup
│   ├── main.js                 ← All UI logic, API calls, state management
│   └── styles.css              ← Glassmorphism design system
│
├── vercel.json                 ← Rewrites all requests to /client/* for Vercel SPA routing
├── requirements.txt            ← All Python dependencies (pinned)
└── docker-compose.yml          ← Docker setup for containerized deployment
```
