"""Database layer — BIBLE §44, ARCHITECTURE.md §2.

SQLite + SQLAlchemy ORM.  Single-user, no User/tenant entities.
Entities: Project, DataSource, DataFile, Dataset, DatasetVersion,
Experiment, TrainingRun, Evaluation, ModelArtifact, Backend, Deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)

from backend.config import settings


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Models  (ARCHITECTURE.md §2 ERD)
# ---------------------------------------------------------------------------

class Project(Base):
    """BIBLE §5 — Project Manager."""

    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    objective = Column(Text, default="")
    problem_type = Column(String, default="")
    status = Column(String, default="created")
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    datasources = relationship("DataSource", back_populates="project")
    datasets = relationship("Dataset", back_populates="project")
    experiments = relationship("Experiment", back_populates="project")
    deployments = relationship("Deployment", back_populates="project")


class DataSource(Base):
    """BIBLE §9 — Data Ingestion Engine."""

    __tablename__ = "datasources"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    original_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    file_type = Column(String, default="")
    size_bytes = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="datasources")
    files = relationship("DataFile", back_populates="datasource")


class DataFile(Base):
    """BIBLE §10 — individual files within a data source."""

    __tablename__ = "datafiles"

    id = Column(String, primary_key=True, default=_uuid)
    datasource_id = Column(String, ForeignKey("datasources.id"), nullable=False)
    path = Column(String, nullable=False)
    detected_type = Column(String, default="")
    schema_json = Column(Text, default="{}")
    stats_json = Column(Text, default="{}")

    datasource = relationship("DataSource", back_populates="files")


class Dataset(Base):
    """BIBLE §13 — Dataset Construction."""

    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    task_type = Column(String, default="")
    created_at = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="datasets")
    versions = relationship("DatasetVersion", back_populates="dataset")


class DatasetVersion(Base):
    """BIBLE §32 — versioned datasets."""

    __tablename__ = "dataset_versions"

    id = Column(String, primary_key=True, default=_uuid)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    version = Column(Integer, default=1)
    path = Column(String, nullable=False)
    num_samples = Column(Integer, default=0)
    split_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=_now)

    dataset = relationship("Dataset", back_populates="versions")


class BackendRow(Base):
    """BIBLE §17 — registered backend adapters."""

    __tablename__ = "backends"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, unique=True, nullable=False)
    adapter_class = Column(String, nullable=False)
    capabilities_json = Column(Text, default="{}")
    license = Column(String, default="")


class Experiment(Base):
    """BIBLE §21 — Experiment Engine."""

    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    model_name = Column(String, nullable=False)
    backend = Column(String, nullable=False)
    config_json = Column(Text, default="{}")
    status = Column(String, default="created")
    created_at = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="experiments")
    training_run = relationship("TrainingRun", back_populates="experiment", uselist=False)
    evaluations = relationship("Evaluation", back_populates="experiment")


class TrainingRun(Base):
    """BIBLE §29 — a single training execution."""

    __tablename__ = "training_runs"

    id = Column(String, primary_key=True, default=_uuid)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    backend = Column(String, nullable=False)
    config_json = Column(Text, default="{}")
    status = Column(String, default="pending")
    pid = Column(String, default="")
    status_file_path = Column(String, default="")
    cost_estimate = Column(Float, default=0.0)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    experiment = relationship("Experiment", back_populates="training_run")
    artifact = relationship("ModelArtifact", back_populates="training_run", uselist=False)


class Evaluation(Base):
    """BIBLE §24-25 — Evaluation Engine + Error Analysis."""

    __tablename__ = "evaluations"

    id = Column(String, primary_key=True, default=_uuid)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    metrics_json = Column(Text, default="{}")
    error_analysis_json = Column(Text, default="{}")
    evaluated_at = Column(DateTime, default=_now)

    experiment = relationship("Experiment", back_populates="evaluations")


class ModelArtifact(Base):
    """BIBLE §31 — Model Registry artifact."""

    __tablename__ = "model_artifacts"

    id = Column(String, primary_key=True, default=_uuid)
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=False)
    path = Column(String, nullable=False)
    model_type = Column(String, default="")
    base_model = Column(String, default="")
    framework = Column(String, default="")
    size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=_now)

    training_run = relationship("TrainingRun", back_populates="artifact")


class Deployment(Base):
    """BIBLE §30 — Deployment Engine."""

    __tablename__ = "deployments"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    model_artifact_id = Column(String, ForeignKey("model_artifacts.id"), nullable=False)
    deploy_type = Column(String, default="")
    endpoint = Column(String, default="")
    status = Column(String, default="created")
    deployed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="deployments")


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


def get_db() -> Session:  # type: ignore[misc]
    """FastAPI dependency — yields a session, closes on exit."""
    db = SessionLocal()
    try:
        yield db  # type: ignore[misc]
    finally:
        db.close()
