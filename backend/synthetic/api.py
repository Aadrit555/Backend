import io
import csv
import json
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .engine import SyntheticEngine, start_generation_job, get_job_status, get_job_records

router = APIRouter(prefix="/api/synthetic", tags=["synthetic"])

class SchemaRequest(BaseModel):
    prompt: str
    rows: int

class GenerateRequest(BaseModel):
    schema_def: Dict[str, Any]

@router.post("/schema")
def generate_schema(req: SchemaRequest):
    try:
        engine = SyntheticEngine()
        schema = engine.plan_dataset(req.prompt, req.rows)
        return {"status": "success", "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
def start_generation(req: GenerateRequest):
    try:
        job_id = start_generation_job(req.schema_def)
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
def check_status(job_id: str):
    status = get_job_status(job_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@router.get("/download/{job_id}/{format}")
def download_dataset(job_id: str, format: str):
    records = get_job_records(job_id)
    if not records:
        raise HTTPException(status_code=404, detail="Job not found or no records generated.")
        
    if format == "json":
        content = json.dumps(records, indent=2)
        return Response(content=content, media_type="application/json", headers={"Content-Disposition": f"attachment; filename=synthetic_dataset.json"})
    elif format == "csv":
        output = io.StringIO()
        if len(records) > 0:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=synthetic_dataset.csv"})
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'csv' or 'json'.")
