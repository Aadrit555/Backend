"""FastAPI application entry point — BIBLE §39, ARCHITECTURE.md §1.

Mounts all routers and initialises the database on startup.
Run with: uvicorn backend.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db import init_db
from backend.gpu_probe import probe_gpus
from backend.status import router as status_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables, ensure storage dirs, probe GPU."""
    init_db()
    for d in (
        settings.raw_dir,
        settings.processed_dir,
        settings.models_dir,
        settings.experiments_dir,
        settings.logs_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    gpus = probe_gpus()
    if gpus:
        print(f"[startup] Detected {len(gpus)} GPU(s): "
              + ", ".join(f"{g.name} ({g.free_mb} MB free)" for g in gpus))
    else:
        print("[startup] No NVIDIA GPU detected — training will be CPU-only.")
    yield


app = FastAPI(
    title="Unified AI/ML Model Builder",
    description="BIBLE §0: Give us your data. Tell us what you want. We'll build the model.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-user local — tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api import router as api_router

# --- Mount routers ---
app.include_router(status_router)
app.include_router(api_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
