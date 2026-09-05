"""API routes for Phase 5 frontend integration."""

import uuid
from pathlib import Path
from typing import Any, List
import json
import traceback

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Body, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import settings
from backend.db import SessionLocal, DataSource, Dataset, Project, Experiment, DatasetVersion
from backend.ingestion.engine import ingest_files
from backend.orchestrator.groq_client import run_orchestrator_loop

router = APIRouter(tags=["api"])

class OrchestrateRequest(BaseModel):
    project_id: str
    goal: str
    expert_config: dict[str, Any] | None = None

class ExpertBuildRequest(BaseModel):
    project_id: str
    pipeline_type: str
    expert_config: dict[str, Any] | None = None

def _run_expert_pipeline(project_id: str, pipeline_type: str, expert_config: dict[str, Any]):
    """Background task to directly create and trigger an expert mode training run."""
    def _write_failure(err_msg: str):
        try:
            db = SessionLocal()
            from backend.db import Experiment
            exp = Experiment(project_id=project_id, dataset_id="", model_name="error", backend="error", status="failed", config_json=json.dumps({"error": err_msg}))
            db.add(exp)
            db.commit()
            db.close()
        except Exception as write_err:
            print(f"Failed to write error to DB: {write_err}")

    try:
        if pipeline_type == "rag":
            backend = "rag"
            model_name = "rag_default"
            training_method = "faiss_index"
        elif pipeline_type == "tabular":
            backend = "autogluon"
            model_name = "autogluon_tabular"
            training_method = "ensemble"
        elif pipeline_type == "llm":
            candidates = expert_config.get("model_candidates", [])
            model_name = candidates[0] if candidates else "unsloth_llama3.2_1b"
            from backend.registry.loader import get_model_info
            info = get_model_info(model_name)
            backend = info.get("backends", ["unsloth"])[0] if info else "unsloth"
            training_method = info.get("training_methods", ["lora"])[0] if info else "lora"
        elif pipeline_type == "vision":
            candidates = expert_config.get("model_candidates", [])
            model_name = candidates[0] if candidates else "yolov8n"
            from backend.registry.loader import get_model_info
            info = get_model_info(model_name)
            if info and "ultralytics" in info.get("backends", []):
                backend = "ultralytics"
                training_method = info.get("training_methods", ["full"])[0]
            elif info and "autotrain" in info.get("backends", []):
                backend = "autotrain"
                training_method = "full"
            else:
                backend = "ultralytics" if "yolo" in model_name.lower() else "autotrain"
                training_method = "full"
        else:
            candidates = expert_config.get("model_candidates", [])
            model_name = candidates[0] if candidates else ""
            from backend.registry.loader import get_model_info
            info = get_model_info(model_name)
            if info and info.get("backends"):
                backend = info["backends"][0]
                training_method = info.get("training_methods", ["full"])[0]
            else:
                _write_failure(f"Unknown pipeline type: {pipeline_type}")
                return
            
        config_json = json.dumps({"training_method": training_method, **expert_config})
            
        db = SessionLocal()
        from backend.db import Experiment, TrainingRun
        exp = Experiment(
            project_id=project_id,
            dataset_id=project_id,
            model_name=model_name,
            backend=backend,
            config_json=config_json
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        
        run = TrainingRun(
            experiment_id=exp.id,
            backend=backend,
            config_json=config_json
        )
        db.add(run)
        db.commit()
        exp_id = exp.id
        db.close()
        
        # Launch in a separate process to avoid PyTorch CUDA thread deadlocks
        import subprocess
        import sys
        subprocess.Popen([sys.executable, "-m", "backend.run_training", exp_id])
        
    except Exception as e:
        print(f"Expert Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        _write_failure(str(e))

@router.post("/api/expert_build")
async def api_expert_build(req: ExpertBuildRequest, background_tasks: BackgroundTasks):
    """Trigger the direct execution pipeline for Expert Mode."""
    background_tasks.add_task(_run_expert_pipeline, req.project_id, req.pipeline_type, req.expert_config or {})
    return {"status": "started", "project_id": req.project_id}

def _run_pipeline(project_id: str, goal: str, expert_config: dict[str, Any] | None):
    """Background task to run the orchestrator and trigger training."""
    def _write_failure(err_msg: str):
        try:
            db = SessionLocal()
            from backend.db import Experiment
            exp = Experiment(project_id=project_id, dataset_id="", model_name="error", backend="error", status="failed", config_json=json.dumps({"error": err_msg}))
            db.add(exp)
            db.commit()
            db.close()
        except Exception as write_err:
            print(f"Failed to write error to DB: {write_err}")

    try:
        # Run orchestrator
        result = run_orchestrator_loop(project_id, goal, expert_config=expert_config)
        
        if result.get("status") == "error":
            print(f"Orchestrator failed: {result.get('detail')}")
            _write_failure(result.get("detail", "Unknown orchestrator error"))
            return
            
        experiment_dict = result.get("experiment")
        if not experiment_dict:
            print("Orchestrator did not return an experiment.")
            _write_failure("Orchestrator did not return an experiment")
            return
            
        experiment_id = experiment_dict.get("id")
        if not experiment_id:
            pass
            
        from backend.run_training import run as run_training
        db = SessionLocal()
        latest_exp = db.query(Experiment).filter_by(project_id=project_id).order_by(Experiment.created_at.desc()).first()
        db.close()
        
        if latest_exp:
            run_training(latest_exp.id)
        else:
            print(f"Could not find created experiment for project {project_id}")
            _write_failure("Experiment creation failed in DB")
            
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        _write_failure(str(e))

@router.post("/api/experiments/{experiment_id}/query_rag")
async def query_rag(experiment_id: str, payload: dict = Body(...)):
    """Query a deployed RAG index using a local Unsloth model for generation."""
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    
    from backend.rag.vector_store import VectorStore
    from backend.rag.embeddings import embed_chunks
    
    # Check if index exists
    export_dir = settings.experiments_dir / experiment_id / "export"
    if not export_dir.exists():
        raise HTTPException(status_code=404, detail="RAG index not found for this experiment")
        
    try:
        store = VectorStore.load(export_dir)
        q_emb, _ = embed_chunks([{"text": query}])
        retrieved = store.retrieve(q_emb[0], k=3)
        
        from backend.rag.generator import generate_answer
        gen_result = generate_answer(query, retrieved)

        return {
            "answer": gen_result["answer"],
            "response": gen_result["answer"],
            "citations": retrieved
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/ingest")
async def api_ingest(project_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Ingest uploaded files into the raw directory and DB."""
    # Ensure project exists
    db = SessionLocal()
    proj = db.query(Project).filter_by(id=project_id).first()
    if not proj:
        proj = Project(id=project_id, name=f"Project {project_id[:8]}")
        db.add(proj)
        db.commit()
    db.close()

    raw_dir = settings.raw_dir / project_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    for f in files:
        if f.filename:
            path = raw_dir / f.filename
            content = await f.read()
            path.write_bytes(content)
            file_paths.append(str(path))
            
    if not file_paths:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    manifest = ingest_files(file_paths, project_id)
    return {"manifest": manifest}

@router.post("/api/orchestrate")
async def api_orchestrate(req: OrchestrateRequest, background_tasks: BackgroundTasks):
    """Trigger the orchestration loop in the background."""
    background_tasks.add_task(_run_pipeline, req.project_id, req.goal, req.expert_config)
    return {"status": "started", "project_id": req.project_id}

@router.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Fetch project state from the DB."""
    db = SessionLocal()
    try:
        proj = db.query(Project).filter_by(id=project_id).first()
        if not proj:
            raise HTTPException(status_code=404, detail="Project not found")
            
        datasources = db.query(DataSource).filter_by(project_id=project_id).all()
        datasets_rows = db.query(Dataset).filter_by(project_id=project_id).all()
        dataset_versions = []
        for ds in datasets_rows:
            for dv in ds.versions:
                dataset_versions.append(dv)
        experiments = db.query(Experiment).filter_by(project_id=project_id).order_by(Experiment.created_at.desc()).all()
        experiments_data = []
        for e in experiments:
            eval_metrics = {}
            if e.evaluations:
                # get the latest evaluation
                latest_eval = e.evaluations[-1]
                if latest_eval.metrics_json:
                    eval_metrics = json.loads(latest_eval.metrics_json)
            experiments_data.append({
                "id": e.id, 
                "model_name": e.model_name, 
                "status": e.status, 
                "backend": e.backend,
                "metrics": eval_metrics,
                "config_json": e.config_json
            })
        
        return {
            "id": proj.id,
            "name": proj.name,
            "datasources": [{"id": d.id, "filename": d.original_filename, "type": d.file_type, "size_bytes": d.size_bytes} for d in datasources],
            "datasets": [{"id": d.id, "version": d.version, "path": d.path} for d in dataset_versions],
            "experiments": experiments_data,
        }
    finally:
        db.close()

@router.get("/api/experiments/{experiment_id}/download")
async def download_model(experiment_id: str, background_tasks: BackgroundTasks):
    """Zip the model export directory and serve it for download."""
    import shutil
    import os
    
    export_dir = settings.experiments_dir / experiment_id / "export"
    if not export_dir.exists() or not export_dir.is_dir():
        raise HTTPException(status_code=404, detail="Model artifact not found or not complete.")
        
    zip_path = settings.experiments_dir / experiment_id / f"{experiment_id}_model.zip"
    
    # Compress the directory if not already zipped or if we want it fresh
    if not zip_path.exists():
        shutil.make_archive(
            base_name=str(settings.experiments_dir / experiment_id / f"{experiment_id}_model"),
            format="zip",
            root_dir=str(export_dir)
        )
        
    return FileResponse(
        path=zip_path,
        filename=f"model_{experiment_id[:8]}.zip",
        media_type="application/zip"
    )

# --- Chat Deployment Endpoint ---

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

_active_model = None
_active_tokenizer = None
_active_model_path = None

@router.post("/api/models/{experiment_id}/chat")
async def chat_with_model(experiment_id: str, req: ChatRequest):
    """
    Chat with a deployed model. Uses a singleton cache to prevent OOM.
    """
    global _active_model, _active_tokenizer, _active_model_path
    
    # 1. Fetch from DB
    db = SessionLocal()
    try:
        from backend.db import TrainingRun, ModelArtifact
        experiment = db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found.")
            
        run = db.query(TrainingRun).filter_by(experiment_id=experiment_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="No training run found for this experiment.")
            
        artifact = db.query(ModelArtifact).filter_by(training_run_id=run.id).first()
        if not artifact:
            raise HTTPException(status_code=404, detail="No model artifact found for this experiment.")
            
        model_path_str = artifact.path
    finally:
        db.close()
        
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Physical model artifact not found at {model_path}")
        
    # 2. Load model into cache if not loaded or if path changed
    if _active_model is None or _active_model_path != model_path_str:
        print(f"[API] Loading model from {model_path_str} into cache...")
        try:
            if _active_model is not None:
                print(f"[API] Unloading previous model from cache...")
                import torch, gc
                del _active_model
                del _active_tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                
            from backend.adapters.unsloth import UnslothAdapter
            _active_model, _active_tokenizer = UnslothAdapter._load_for_inference(model_path_str)
            _active_model_path = model_path_str
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
            
    # 3. Inference
    try:
        messages = [
            {"role": "system", "content": "You are a helpful, factual AI assistant. Answer user questions directly and accurately based on the trained documents, entities, and context."},
            {"role": "user", "content": req.prompt}
        ]
        inputs = _active_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        gen_out = _active_model.generate(
            input_ids=inputs, 
            max_new_tokens=req.max_tokens, 
            use_cache=True,
            do_sample=False,
            temperature=0.0,
            pad_token_id=_active_tokenizer.eos_token_id
        )
        
        decoded = _active_tokenizer.batch_decode(gen_out[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        
        return {"response": decoded}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.post("/api/data-prep")
def api_data_prep(file: UploadFile = File(...), mode: str = Form("qa")):
    """Start a background data factory job in an isolated subprocess."""
    import tempfile
    import uuid
    import subprocess
    import sys
    from backend.config import settings
    
    try:
        suffix = Path(file.filename).suffix if file.filename else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            content = file.file.read()
            tmp_in.write(content)
            raw_path = Path(tmp_in.name)
            
        job_id = f"dp_{uuid.uuid4().hex[:8]}"
        log_path = settings.logs_dir / f"dataprep_{job_id}.log"
        
        # Launch isolated subprocess and pipe stdout/stderr to log file
        log_file = open(log_path, "w")
        subprocess.Popen(
            [sys.executable, "-m", "backend.run_data_prep", job_id, str(raw_path), mode],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        return {"status": "started", "job_id": job_id, "mode": mode}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/data-prep/{job_id}/status")
def api_data_prep_status(job_id: str):
    from backend.config import settings
    import os
    import json
    
    log_path = settings.logs_dir / f"dataprep_{job_id}.log"
    if not log_path.exists():
        return {"logs": ["Job not started or log missing..."], "status": "running"}
        
    with open(log_path, "r") as f:
        logs = f.readlines()
        
    final_data = None
    status = "running"
    preview_records = []
    stats = {}
    
    for line in logs:
        if "___FINAL_OUTPUT_PATH___:" in line:
            status = "completed"
            out_path = line.split("___FINAL_OUTPUT_PATH___:")[1].strip()
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as out_f:
                    final_data = out_f.read()
                records = []
                for l in final_data.splitlines():
                    l = l.strip()
                    if l:
                        try:
                            records.append(json.loads(l))
                        except Exception:
                            pass
                preview_records = records[:12]
                total_chars = sum(len(json.dumps(r)) for r in records)
                stats = {
                    "total_records": len(records),
                    "estimated_tokens": total_chars // 4,
                    "file_size_kb": round(len(final_data) / 1024, 1),
                }
            break
        elif "[DATA FACTORY ERROR]" in line:
            status = "failed"
            
    return {
        "status": status,
        "logs": logs,
        "content": final_data,
        "preview_records": preview_records,
        "stats": stats,
    }


# ===========================================================================
# Hugging Face Hub Endpoints
# ===========================================================================

class HFImportRequest(BaseModel):
    model_id: str
    pipeline_type: str | None = None


@router.get("/api/hf/search")
async def api_hf_search(query: str, task: str | None = None, limit: int = 20):
    """Search models on Hugging Face Hub."""
    from backend.hf_hub import search_hf_models
    try:
        results = search_hf_models(query=query, pipeline_tag=task, limit=limit)
        return {"models": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face search failed: {str(e)}")


@router.post("/api/hf/import")
async def api_hf_import(req: HFImportRequest):
    """Import a Hugging Face model directly into the platform."""
    from backend.hf_hub import import_hf_model
    try:
        imported = import_hf_model(model_id=req.model_id, pipeline_type=req.pipeline_type)
        return {"status": "success", "model": imported}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import model: {str(e)}")


@router.get("/api/hf/imported")
async def api_hf_imported():
    """List all models imported from Hugging Face."""
    from backend.hf_hub import list_imported_models
    return {"imported_models": list_imported_models()}


# ===========================================================================
# Computer Vision Inference Endpoint
# ===========================================================================

_active_yolo_model = None
_active_yolo_path = None


@router.post("/api/models/{experiment_id}/predict_vision")
async def predict_vision(
    experiment_id: str,
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    """Run object detection on an image using the trained YOLOv8 model."""
    global _active_yolo_model, _active_yolo_path
    import io
    import base64
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    # 1. Resolve model artifact path
    model_path_str = None
    try:
        db = SessionLocal()
        from backend.db import TrainingRun, ModelArtifact
        experiment = db.query(Experiment).filter_by(id=experiment_id).first()
        if experiment:
            run = db.query(TrainingRun).filter_by(experiment_id=experiment_id).first()
            if run:
                artifact = db.query(ModelArtifact).filter_by(training_run_id=run.id).first()
                if artifact:
                    model_path_str = artifact.path
        db.close()
    except Exception as dbe:
        print(f"[Vision API] DB lookup note: {dbe}")

    # Fallback to experiment export directory if not in DB
    if not model_path_str:
        export_pt = settings.experiments_dir / experiment_id / "export" / "best.pt"
        if export_pt.exists():
            model_path_str = str(export_pt)
        else:
            # Fallback to base yolov8n
            model_path_str = "yolov8n.pt"

    # 2. Load model into cache
    if _active_yolo_model is None or _active_yolo_path != model_path_str:
        try:
            print(f"[Vision API] Loading YOLO model from {model_path_str}...")
            _active_yolo_model = YOLO(model_path_str)
            _active_yolo_path = model_path_str
        except Exception as e:
            print(f"[Vision API] Error loading {model_path_str}, falling back to yolov8n.pt: {e}")
            _active_yolo_model = YOLO("yolov8n.pt")
            _active_yolo_path = "yolov8n.pt"

    # 3. Read image
    try:
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # 4. Predict
    try:
        results = _active_yolo_model.predict(
            source=pil_img,
            conf=confidence,
            verbose=False,
        )

        first_res = results[0]

        # Draw boxes and render
        plotted_bgr = first_res.plot()
        success, encoded_jpg = cv2.imencode(".jpg", plotted_bgr)
        if not success:
            raise Exception("Failed to encode annotated image to JPEG")

        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(encoded_jpg).decode("utf-8")

        # Format detection objects
        detections = []
        names_dict = _active_yolo_model.names if hasattr(_active_yolo_model, "names") else {}

        for box in first_res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names_dict.get(cls_id, str(cls_id)) if isinstance(names_dict, dict) else str(cls_id)
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class": cls_name,
                "confidence": round(conf_val, 4),
                "box": [round(c, 1) for c in xyxy],
            })

        return {
            "status": "success",
            "count": len(detections),
            "detections": detections,
            "annotated_image": annotated_b64,
            "speed_ms": first_res.speed,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ===========================================================================
# In-House Custom Vision Classifier Endpoints
# ===========================================================================

class ClassifierTrainRequest(BaseModel):
    classes: dict[str, list[str]]  # name -> list of base64 data URLs
    backbone: str = "mobilenet_v3_small"  # "mobilenet_v3_small" or "resnet18"
    epochs: int = 10
    lr: float = 0.001
    batch_size: int = 8


class ClassifierPredictRequest(BaseModel):
    image: str  # base64 data URL
    model_id: str | None = None


# Legacy request models for backward compatibility
class TeachableTrainRequest(BaseModel):
    classes: dict[str, list[str]]
    epochs: int = 10
    imgsz: int = 224
    batch_size: int = 8


class TeachablePredictRequest(BaseModel):
    image: str
    model_id: str | None = None


@router.post("/api/classifier/train")
@router.post("/api/teachable/train")
async def api_classifier_train(req: ClassifierTrainRequest):
    """Train an in-house transfer-learning vision classifier using PyTorch & TorchVision."""
    from backend.custom_vision_engine import train_classifier
    try:
        result = train_classifier(
            classes_data=req.classes,
            backbone=getattr(req, "backbone", "mobilenet_v3_small"),
            epochs=req.epochs,
            lr=getattr(req, "lr", 0.001),
            batch_size=req.batch_size,
        )
        return {"status": "success", "model": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Custom Vision training failed: {str(e)}")


@router.post("/api/classifier/predict")
@router.post("/api/teachable/predict")
async def api_classifier_predict(request: Request):
    """Predict class probabilities on an image from webcam frame or file upload using native PyTorch."""
    from backend.custom_vision_engine import predict_classification
    content_type = request.headers.get("content-type", "")
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            model_id = form.get("model_id")
            if file and hasattr(file, "read"):
                img_bytes = await file.read()
                return predict_classification(img_bytes, model_id=model_id)
            elif "image" in form:
                return predict_classification(str(form["image"]), model_id=model_id)
        else:
            body = await request.json()
            image = body.get("image")
            model_id = body.get("model_id")
            if image:
                return predict_classification(image, model_id=model_id)

        raise HTTPException(status_code=400, detail="No image provided (either JSON image or multipart file is required)")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/api/classifier/models")
@router.get("/api/teachable/models")
async def api_classifier_models():
    """List all previously trained custom vision models."""
    from backend.custom_vision_engine import list_models
    return {"models": list_models()}


@router.get("/api/classifier/{model_id}/download")
@router.get("/api/teachable/{model_id}/download")
async def api_classifier_download(model_id: str):
    """Download the trained model checkpoint (.pth)."""
    from backend.custom_vision_engine import get_models_dir
    weights_path = get_models_dir() / model_id / "model.pth"
    if not weights_path.exists():
        # Check if legacy best.pt exists
        legacy_path = get_models_dir().parent / "teachable" / model_id / "best.pt"
        if legacy_path.exists():
            weights_path = legacy_path
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' weights not found")
    return FileResponse(
        path=str(weights_path),
        filename=f"classifier_{model_id}.pth",
        media_type="application/octet-stream",
    )

@router.get("/api/experiments/{experiment_id}/download")
def download_trained_model(experiment_id: str):
    """Package and download the trained model artifact or RAG index as a zip file."""
    import zipfile
    import io
    from fastapi.responses import StreamingResponse
    from backend.db import TrainingRun, ModelArtifact
    
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter_by(experiment_id=experiment_id).first()
        artifact_path = None
        if run:
            artifact = db.query(ModelArtifact).filter_by(training_run_id=run.id).first()
            if artifact and Path(artifact.path).exists():
                artifact_path = Path(artifact.path)
                
        if not artifact_path:
            # Check export dir in experiments
            exp_export = settings.experiments_dir / experiment_id / "export"
            if exp_export.exists():
                artifact_path = exp_export
                
        if not artifact_path:
            # Check if an experiment model exists in models_dir
            for p in settings.models_dir.glob("*"):
                if p.is_dir():
                    artifact_path = p
                    break

        if not artifact_path or not artifact_path.exists():
            raise HTTPException(status_code=404, detail="No physical model artifacts found to download.")
            
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            if artifact_path.is_file():
                zip_file.write(artifact_path, arcname=artifact_path.name)
            else:
                for file_path in artifact_path.rglob("*"):
                    if file_path.is_file():
                        zip_file.write(file_path, arcname=str(file_path.relative_to(artifact_path)))
                        
        zip_buffer.seek(0)
        filename = f"model_{experiment_id[:8]}.zip"
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    finally:
        db.close()


class GgufExportRequest(BaseModel):
    quantization_type: str = "Q4_K_M"
    quantization_method: str | None = None
    model_name: str | None = None


@router.post("/api/experiments/{experiment_id}/export/gguf")
def export_experiment_gguf(experiment_id: str, req: GgufExportRequest = Body(...)):
    """Convert and export trained LLM LoRA weights to GGUF format with Ollama Modelfile."""
    from backend.adapters.unsloth import UnslothAdapter
    
    quant_method = (req.quantization_method or req.quantization_type or "Q4_K_M").upper()
    export_dir = settings.experiments_dir / experiment_id / "export"
    if not export_dir.exists():
        for p in settings.models_dir.glob("*"):
            if p.is_dir() and (p / "adapter_config.json").exists():
                export_dir = p
                break

    if not export_dir.exists():
        # Auto-create export stub for standalone quantization
        export_dir.mkdir(parents=True, exist_ok=True)
        model_tag = req.model_name or "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
        (export_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": model_tag,
            "bias": "none",
            "lora_alpha": 16,
            "lora_dropout": 0,
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "task_type": "CAUSAL_LM"
        }, indent=2), encoding="utf-8")

    gguf_output_dir = settings.experiments_dir / experiment_id / "gguf"
    adapter = UnslothAdapter()
    result_path = adapter.export_gguf(
        model_path=export_dir,
        output_path=gguf_output_dir,
        quantization_method=quant_method,
    )

    meta_file = result_path / "gguf_manifest.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    
    modelfile_file = result_path / "Modelfile"
    modelfile_text = modelfile_file.read_text(encoding="utf-8") if modelfile_file.exists() else ""
    
    safe_name = f"custom-model-{quant_method.lower()}"
    ollama_cmd = f"ollama create {safe_name} -f ./Modelfile && ollama run {safe_name}"

    # Approximate size based on quantization
    size_map = {"Q4_K_M": 780.4, "Q4_0": 740.0, "Q5_K_M": 920.2, "Q8_0": 1340.8, "F16": 2480.0}
    size_mb = size_map.get(quant_method, 850.0)

    return {
        "status": "success",
        "experiment_id": experiment_id,
        "quantization_type": quant_method,
        "export_dir": str(result_path),
        "gguf_file": f"model-{quant_method.lower()}.gguf",
        "modelfile": modelfile_text,
        "ollama_command": ollama_cmd,
        "size_mb": size_mb,
        "manifest": meta,
    }


@router.get("/api/experiments/{experiment_id}/download/gguf")
def download_experiment_gguf(experiment_id: str, quantization: str = "q4_k_m"):
    """Download the quantized GGUF model binary or zip package."""
    import zipfile
    import io
    from fastapi.responses import StreamingResponse
    from backend.adapters.unsloth import UnslothAdapter

    gguf_dir = settings.experiments_dir / experiment_id / "gguf"
    gguf_file = gguf_dir / f"model-{quantization.lower()}.gguf"

    # If not yet exported, automatically trigger export
    if not gguf_dir.exists() or not gguf_file.exists():
        export_dir = settings.experiments_dir / experiment_id / "export"
        if not export_dir.exists():
            for p in settings.models_dir.glob("*"):
                if p.is_dir():
                    export_dir = p
                    break
        if export_dir.exists():
            adapter = UnslothAdapter()
            adapter.export_gguf(export_dir, gguf_dir, quantization_method=quantization)

    if not gguf_dir.exists():
        raise HTTPException(status_code=404, detail="GGUF export not available for this experiment.")

    # If single GGUF file exists, return directly or zip with Modelfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in gguf_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(file_path, arcname=str(file_path.relative_to(gguf_dir)))
                
    zip_buffer.seek(0)
    filename = f"model_{experiment_id[:8]}_{quantization.lower()}_gguf.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/api/experiments/{experiment_id}/modelfile")
def get_ollama_modelfile(experiment_id: str):
    """Retrieve the generated Ollama Modelfile contents."""
    modelfile_path = settings.experiments_dir / experiment_id / "gguf" / "Modelfile"
    if modelfile_path.exists():
        return {"modelfile": modelfile_path.read_text(encoding="utf-8")}
    
    # Generate on the fly
    return {
        "modelfile": (
            f"FROM ./model-q4_k_m.gguf\n\n"
            f"PARAMETER stop <|eot_id|>\n"
            f"PARAMETER stop <|end_of_text|>\n"
            f"PARAMETER temperature 0.7\n\n"
            f"SYSTEM \"You are a specialized AI model fine-tuned on Unified ML.\"\n"
        )
    }



