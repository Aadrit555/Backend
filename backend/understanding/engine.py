"""Data Understanding Engine — BIBLE §10-11, ARCHITECTURE.md §1.

Analyses ingested data sources to detect:
  - Column types, schemas, metadata
  - Missing values, duplicates, outliers
  - Potential target columns
  - Relationships / join keys across data sources (the "Data Map")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import settings


def _analyze_tabular(path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    
    row_count = len(df)
    columns = []
    
    likely_id_column = None
    likely_target_column = None
    
    best_id_cardinality = 0
    best_target_cardinality = row_count + 1
    
    for col in df.columns:
        col_series = df[col]
        dtype = str(col_series.dtype)
        null_count = int(col_series.isnull().sum())
        cardinality = int(col_series.nunique(dropna=True))
        
        # ID heuristic: High cardinality, ideally unique, no nulls
        # Target heuristic: Low cardinality categorical or numeric, no nulls
        
        is_unique = (cardinality == row_count) and (null_count == 0)
        is_low_cardinality = (cardinality >= 2) and (cardinality <= 50)
        
        if is_unique:
            # We just take the first unique one as ID
            if likely_id_column is None:
                likely_id_column = col
        
        if null_count == 0 and is_low_cardinality:
            # Pick the lowest cardinality valid column as target
            if cardinality < best_target_cardinality:
                best_target_cardinality = cardinality
                likely_target_column = col
                
        columns.append({
            "name": col,
            "dtype": dtype,
            "null_count": null_count,
            "cardinality": cardinality,
        })
        
    return {
        "type": "tabular",
        "row_count": row_count,
        "columns": columns,
        "likely_id_column": likely_id_column,
        "likely_target_column": likely_target_column,
    }


def _analyze_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        # We might have JSON lines or standard JSON. We'll try to parse the whole file first.
        # If it fails, we fall back to line by line.
        try:
            f.seek(0)
            data = json.load(f)
        except Exception:
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                if len(data) > 100:  # just limit for parsing schema
                    break
                    
    structure = "list" if isinstance(data, list) else "object"
    schema = {}
    
    if structure == "list" and len(data) > 0 and isinstance(data[0], dict):
        # Infer schema from first 10 records
        for record in data[:10]:
            if isinstance(record, dict):
                for k, v in record.items():
                    if k not in schema:
                        schema[k] = type(v).__name__
                    
    return {
        "type": "json",
        "structure": structure,
        "inferred_schema": schema,
    }


def _analyze_text(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read(10000) # Read chunk for analysis
        
    word_count = len(content.split())
    lines = content.splitlines()
    
    # Naive "looks tabular" check: many lines have the exact same number of delimiter-separated tokens
    looks_tabular = False
    if len(lines) > 5:
        delimiters = [",", "\t", "|"]
        for delim in delimiters:
            counts = [len(line.split(delim)) for line in lines[:10] if line.strip()]
            if len(counts) > 3 and len(set(counts)) == 1 and counts[0] > 1:
                looks_tabular = True
                break
                
    return {
        "type": "text",
        "word_count_estimate": word_count, # roughly from first 10k bytes
        "looks_tabular": looks_tabular,
    }


def _analyze_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            return {
                "type": "image",
                "format": img.format,
                "dimensions": img.size,
                "file_size": path.stat().st_size,
            }
    except Exception:
        return {
            "type": "image",
            "file_size": path.stat().st_size,
            "error": "Could not parse image dimensions",
        }


def analyze_file(path: Path, file_type: str) -> dict[str, Any]:
    """Analyze a single file based on its detected type."""
    if not path.exists():
        return {"error": "File not found"}
        
    if file_type in ["csv", "xlsx"]:
        return _analyze_tabular(path)
    elif file_type == "json":
        return _analyze_json(path)
    elif file_type in ["txt", "pdf"]:
        return _analyze_text(path)
    elif file_type == "image":
        return _analyze_image(path)
    else:
        return {
            "type": file_type,
            "file_size": path.stat().st_size,
            "status": "unsupported for deep analysis"
        }


def analyze_project(project_id: str) -> dict[str, Any]:
    """Run analyze_file over every DataSource in the project."""
    from backend.db import SessionLocal, DataSource
    from backend.ingestion.engine import detect_file_type
    
    report = {
        "project_id": project_id,
        "file_type_counts": {},
        "sources": []
    }
    
    db = SessionLocal()
    try:
        datasources = db.query(DataSource).filter_by(project_id=project_id).all()
        for ds in datasources:
            path = Path(ds.stored_path)
            if path.exists() and path.is_file():
                ftype = ds.file_type or detect_file_type(path)
                report["file_type_counts"][ftype] = report["file_type_counts"].get(ftype, 0) + 1
                
                analysis = analyze_file(path, ftype)
                
                # Combine DB metadata with the deep analysis
                source_info = {
                    "id": ds.id,
                    "filename": ds.original_filename,
                    "file_type": ftype,
                    "size_bytes": ds.size_bytes,
                }
                source_info.update(analysis)
                report["sources"].append(source_info)
    finally:
        db.close()
            
    return report


def extract_document(path: Path) -> list[dict[str, Any]]:
    """Extract text and metadata (page/section) from a document for RAG."""
    from backend.ingestion.engine import detect_file_type
    ftype = detect_file_type(path)
    
    chunks = []
    
    if ftype == "pdf":
        try:
            import pypdf
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks.append({
                            "text": text.strip(),
                            "metadata": {
                                "source": path.name,
                                "page": i + 1
                            }
                        })
        except Exception as e:
            print(f"Error extracting PDF {path}: {e}")
            
    elif ftype == "docx":
        try:
            import docx
            doc = docx.Document(path)
            text = "\\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if text:
                chunks.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": path.name
                    }
                })
        except Exception as e:
            print(f"Error extracting DOCX {path}: {e}")
            
    elif ftype in ["txt", "md"]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                if text.strip():
                    chunks.append({
                        "text": text.strip(),
                        "metadata": {
                            "source": path.name
                        }
                    })
        except Exception as e:
            print(f"Error extracting TXT {path}: {e}")
            
    elif ftype == "json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                # Try JSONL format first
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if text:
                            chunks.append({
                                "text": text.strip(),
                                "metadata": record.get("metadata", {"source": path.name})
                            })
                    except json.JSONDecodeError:
                        # Fallback to standard JSON array if first line isn't JSONL
                        f.seek(0)
                        data = json.load(f)
                        if isinstance(data, list):
                            for record in data:
                                if isinstance(record, dict) and record.get("text"):
                                    chunks.append({
                                        "text": record["text"].strip(),
                                        "metadata": record.get("metadata", {"source": path.name})
                                    })
                        break
        except Exception as e:
            print(f"Error extracting JSON {path}: {e}")
            
    return chunks

def detect_relationships(project_id: str) -> dict[str, Any]:
    """Find shared column names across all tabular files in a project.
    
    Returns a graph structure: {nodes: [{id, filename}], edges: [{from, to, shared_column}]}
    """
    raw_dir = settings.raw_dir / project_id
    if not raw_dir.exists():
        return {"nodes": [], "edges": []}
        
    from backend.ingestion.engine import detect_file_type
    
    nodes = []
    schemas = {}
    
    # 1. Gather nodes and column sets
    for path in raw_dir.iterdir():
        if path.is_file():
            ftype = detect_file_type(path)
            if ftype in ["csv", "xlsx"]:
                analysis = _analyze_tabular(path)
                col_names = {col["name"] for col in analysis.get("columns", [])}
                
                node_id = path.name
                nodes.append({"id": node_id, "filename": path.name})
                schemas[node_id] = col_names
                
    edges = []
    
    # 2. Find intersections
    node_ids = list(schemas.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            id1 = node_ids[i]
            id2 = node_ids[j]
            shared = schemas[id1].intersection(schemas[id2])
            
            for col in shared:
                edges.append({
                    "from": id1,
                    "to": id2,
                    "shared_column": col
                })
                
    return {
        "nodes": nodes,
        "edges": edges
    }
