# Unified AI Platform: Project Status

## What We Have Achieved
- **End-to-End Orchestrator (The "Brain"):** Built a robust orchestration loop using Groq (GPT OSS 120B) that plans machine learning pipelines from natural language goals (Discovery -> Formulation -> Dataset -> Execution -> Evaluation -> Deployment).
- **World 1 (LLMs):** Integrated `Unsloth` for hyper-efficient 4-bit LoRA/QLoRA fine-tuning. We built a strict VRAM diagnostic pipeline and successfully verified that `Qwen2.5-3B`, `LLaMA-3.2-1B`, and `LLaMA-3.2-3B` can be trained entirely within a 5.66GB VRAM constraint.
- **World 2 (Tabular):** Integrated `AutoGluon` to automatically train and select the best ensemble of XGBoost, LightGBM, and Random Forest models on CPU.
- **World 4 (RAG):** Built a complete Retrieval-Augmented Generation pipeline using FAISS vector indexing and local embeddings for document-grounded query answering.
- **Frontend UI (Phase 5):** Developed a Next.js application with two distinct views:
  - **Beginner Mode:** A simplified interface where users drag-and-drop data, state their goal, and the orchestrator automatically picks the best model (now with an optional dropdown).
  - **Expert Mode:** A power-user interface exposing raw hyperparameters, exact model selection, evaluation metrics, and time limits.
- **Real-Time API & Deployment:** Implemented a unified FastAPI backend that seamlessly deploys trained models (both AutoGluon predictors and Unsloth LLMs) and exposes them through a chat/prediction interface in the frontend.

## What We Are Using
- **Core Backend:** Python, FastAPI, SQLAlchemy, Uvicorn.
- **LLM Intelligence:** Groq API (tool-calling for orchestration).
- **Machine Learning / AI Adapters:** 
  - `Unsloth` (for memory-efficient LLM fine-tuning).
  - `AutoGluon` (for Tabular AutoML).
  - `FAISS` & `sentence-transformers` (for RAG).
  - `Ultralytics` / YOLOv8 (for Vision).
- **Frontend:** Next.js (React), standard CSS, `lucide-react` for icons.
- **Hardware Constraint:** Actively developing against a strict 5.66GB usable VRAM limit (NVIDIA RTX 3050 Laptop GPU).

## How We Have Achieved It
- **Modular Adapter Pattern:** Every AI capability (Unsloth, AutoGluon, RAG) inherits from a common `BackendAdapter` interface (`prepare`, `train`, `evaluate`, `deploy`). This abstracts the complexity away from the API.
- **Centralized Capability Registry:** We maintain a `capabilities.yaml` file that acts as the source of truth for what the platform *can* do, including strict VRAM estimates (`min_mb`, `recommended_mb`). 
- **Strict Validation Gates:** Before the orchestrator is allowed to execute a pipeline, it hits a validation gate that queries `nvidia-smi` to ensure the requested model mathematically fits on the available GPU.
- **Hardware Diagnostics:** We wrote standalone diagnostic scripts (like `model_diagnostics.py`) to test real physical limits (e.g. tracking `torch.cuda.max_memory_reserved()`) before allowing models into the registry. 

## What Is Yet To Be Achieved
- **World 3 (Vision) Full Integration:** While the registry knows about YOLOv8, the end-to-end flow (UI -> Orchestrator -> Ultralytics Adapter -> Deployment) needs final polishing and E2E verification.
- **True Side-by-Side Model Comparison:** We skipped the feature to train multiple LLMs in parallel (or sequentially) and compare them side-by-side in the UI due to the heavy time cost (10-15+ minutes per model). 
- **Chat Interface Streaming:** The current LLM and RAG chat interface blocks until the full response is generated. We need to implement server-sent events (SSE) for token-by-token streaming.
- **Multi-Tenant Architecture:** The current implementation assumes a local, single-user environment. Moving to production requires user authentication, secure workspace isolation, and robust background job queueing (e.g., Celery/Redis).
- **Advanced Export:** Allowing users to seamlessly export Unsloth models to GGUF format for Ollama/LM Studio execution directly from the UI.
