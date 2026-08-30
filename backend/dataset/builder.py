"""Dataset Builder — BIBLE §12-13, ARCHITECTURE.md §1.

Two distinct responsibilities:

1. Dataset construction (§13) — build a training dataset that may not
   have existed before from raw data sources.
2. Dataset cleaning (§12) — dedup, handle missing values, fix encodings,
   detect outliers, rebalance classes.  Every transformation is recorded
   as an audit log entry ("never silently destroy data").

Versioned output is written to storage/processed/{dataset_id}/v{N}/.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session


def create_dataset(
    db: Session,
    project_id: str,
    datasource_ids: list[str] | str,
    task_type: str,
    target_column: str | None = None,
    instructions: str | None = None,
) -> dict[str, Any]:
    """Construct a training dataset from one or more data sources.

    Backs the create_dataset tool (ARCHITECTURE.md §4).
    """
    import json
    import shutil
    from pathlib import Path
    from backend.config import settings
    from backend.db import DataSource, DataFile, Dataset, DatasetVersion
    
    if isinstance(datasource_ids, str):
        ds_ids = [did.strip() for did in datasource_ids.split(",") if did.strip()]
    else:
        ds_ids = datasource_ids
        
    if not ds_ids:
        return {"error": "invalid_input", "detail": "No datasource_ids provided."}

    first_ds_id = ds_ids[0]
    
    datasource = db.query(DataSource).filter_by(id=first_ds_id, project_id=project_id).first()
    if not datasource:
        return {"error": "not_found", "detail": f"DataSource {first_ds_id} not found."}
        
    datafile = db.query(DataFile).filter_by(datasource_id=datasource.id).first()
    if not datafile:
        return {"error": "not_found", "detail": f"No DataFile found for DataSource {first_ds_id}."}

    dataset = Dataset(
        project_id=project_id,
        name=f"Dataset from {datasource.original_filename}",
        task_type=task_type
    )
    db.add(dataset)
    db.flush()
    
    version_dir = settings.processed_dir / project_id / dataset.id / "v1"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(datafile.path)
    dest_path = version_dir / source_path.name
    shutil.copy2(source_path, dest_path)
    
    num_samples = 0
    try:
        if datafile.stats_json:
            stats = json.loads(datafile.stats_json)
            num_samples = stats.get("num_rows", 0)
    except Exception:
        pass
        
    split_info = {"train": str(dest_path), "target_column": target_column or ""}
    
    dataset_version = DatasetVersion(
        dataset_id=dataset.id,
        version=1,
        path=str(dest_path),
        num_samples=num_samples,
        split_json=json.dumps(split_info)
    )
    db.add(dataset_version)
    db.commit()
    db.refresh(dataset_version)
    
    return {
        "status": "created",
        "dataset_id": dataset.id,
        "dataset_version_id": dataset_version.id,
        "path": dataset_version.path,
        "num_samples": num_samples,
        "message": "Dataset and initial version created successfully."
    }


def clean_dataset(
    db: Session,
    dataset_version_id: str,
    operations: list[str] | None = None,
) -> dict[str, Any]:
    """Run automated cleaning on a dataset version.

    Backs the clean_dataset tool.
    Every operation is recorded in an audit log.
    """
    # TODO (Phase 1)
    raise NotImplementedError


def improve_dataset(
    db: Session,
    dataset_version_id: str,
    strategy: str,
    instructions: str | None = None,
) -> dict[str, Any]:
    """Apply a targeted improvement: augment, rebalance, denoise, synthetic, relabel.

    Backs the improve_dataset tool.
    """
    # TODO (Phase 1)
    raise NotImplementedError
