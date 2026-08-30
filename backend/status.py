"""Training status endpoint — ARCHITECTURE.md §7.

Exposes the status.json written by run_training.py:
  - GET  /api/experiments/{experiment_id}/status  (polling)
  - WS   /ws/experiments/{experiment_id}/status   (live push)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings

router = APIRouter(tags=["status"])


def _read_status(experiment_id: str) -> dict:
    path = settings.experiments_dir / experiment_id / "status.json"
    if not path.exists():
        return {"stage": "unknown", "pct": 0, "message": "No status file found.", "log": []}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        # New array format — return latest entry as current + full log
        current = data[-1] if data else {"stage": "unknown", "pct": 0, "message": "Empty log."}
        return {**current, "log": data}
    else:
        # Old single-object format — wrap in a list
        return {**data, "log": [data]}


@router.get("/api/experiments/{experiment_id}/status")
async def get_status(experiment_id: str) -> dict:
    """Poll the current training status."""
    return _read_status(experiment_id)


@router.websocket("/ws/experiments/{experiment_id}/status")
async def ws_status(websocket: WebSocket, experiment_id: str) -> None:
    """Push status updates whenever status.json changes."""
    await websocket.accept()
    last = ""
    try:
        while True:
            current = json.dumps(_read_status(experiment_id))
            if current != last:
                await websocket.send_text(current)
                last = current
                status = json.loads(current)
                if status.get("stage") in ("completed", "failed"):
                    break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
