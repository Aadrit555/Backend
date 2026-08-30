"""Data Ingestion Engine — BIBLE §9-10, ARCHITECTURE.md §1.

Handles file uploads and type detection.  Uploaded files are stored
immutably under storage/raw/{project_id}/ and registered as DataSource
rows in the database.

Supports: CSV, Excel, JSON, PDF, TXT, images, audio, video, Parquet, ZIP.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from backend.config import settings


def detect_file_type(path: Path) -> str:
    """Detect the type of a single file by extension + quick content sniff.

    Returns a normalised type string: csv, xlsx, json, pdf, txt,
    image, audio, video, parquet, zip, unknown.
    """
    if not path.exists():
        return "unknown"

    suffix = path.suffix.lower()
    
    # 1. Image, Audio, Video, Zip usually rely on extensions or magic bytes.
    # For MVP, we will rely on extension for binary formats and just sniff text formats.
    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]:
        return "image"
    if suffix in [".mp3", ".wav", ".flac", ".m4a"]:
        return "audio"
    if suffix in [".mp4", ".avi", ".mov", ".mkv"]:
        return "video"
    if suffix in [".zip", ".tar", ".gz"]:
        return "zip"
    if suffix in [".pdf"]:
        return "pdf"
    if suffix in [".xlsx", ".xls"]:
        return "xlsx"
    if suffix in [".parquet"]:
        return "parquet"

    # 2. Sniff text formats to ensure they are what they say they are.
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_chars = f.read(1024).strip()
            
            # Empty file
            if not first_chars:
                return "txt"
                
            # JSON Sniff
            if suffix in [".json", ".jsonl"]:
                try:
                    # Try to parse the snippet or first line
                    if first_chars.startswith("[") or first_chars.startswith("{"):
                        json.loads(first_chars + "]" if first_chars.startswith("[") else first_chars + "}")
                        return "json"
                except Exception:
                    # Might be valid jsonl, let's just check if it parses line by line
                    f.seek(0)
                    try:
                        first_line = f.readline()
                        json.loads(first_line)
                        return "json"
                    except Exception:
                        pass
                return "txt" # fallback
                
            # CSV Sniff
            if suffix == ".csv":
                import csv
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    if dialect.delimiter in [",", ";", "\t", "|"]:
                        return "csv"
                except csv.Error:
                    pass
                return "txt" # fallback
            
            # Default to text if we can read it as text
            return "txt"
            
    except UnicodeDecodeError:
        # It's binary, return unknown if it didn't match earlier
        return "unknown"


def ingest_files(source_paths: list[str], project_id: str) -> list[dict[str, Any]]:
    """Copy files to storage/raw/<project_id>/ unmodified.

    Returns a manifest of ingested files containing id, filename, file_type, size_bytes, and sha256.
    """
    from backend.db import SessionLocal, DataSource, DataFile
    
    raw_dir = settings.raw_dir / project_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    
    db = SessionLocal()
    
    try:
        for sp in source_paths:
            source_path = Path(sp)
            if not source_path.exists() or not source_path.is_file():
                continue
                
            filename = source_path.name
            dest_path = raw_dir / filename
            
            # Never mutate original, copy to storage (skip if already there)
            if source_path.resolve() != dest_path.resolve():
                shutil.copy2(source_path, dest_path)
            
            # Calculate size and sha256
            size_bytes = dest_path.stat().st_size
            
            sha256_hash = hashlib.sha256()
            with open(dest_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
                    
            file_type = detect_file_type(dest_path)
            
            # Create DataSource
            ds = DataSource(
                project_id=project_id,
                original_filename=filename,
                stored_path=str(dest_path),
                file_type=file_type,
                size_bytes=size_bytes
            )
            db.add(ds)
            db.flush() # flush to get ds.id
            
            # Create DataFile
            df = DataFile(
                datasource_id=ds.id,
                path=str(dest_path),
                detected_type=file_type
            )
            db.add(df)
            db.commit()
            
            manifest.append({
                "id": ds.id,
                "filename": filename,
                "file_type": file_type,
                "size_bytes": size_bytes,
                "sha256": sha256_hash.hexdigest(),
            })
    finally:
        db.close()

    return manifest
