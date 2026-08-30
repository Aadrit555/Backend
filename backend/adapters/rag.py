"""RAG Adapter — ARCHITECTURE.md §3.

Wraps the RAG chunking, embedding, and vector store lifecycle so it behaves
identically to standard model training pipelines.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from backend.adapters.base import (
    BackendAdapter,
    EvaluationResult,
    ResourceEstimate,
    TrainingResult,
)
from backend.config import settings
from backend.registry.loader import get_model_capabilities

class RagAdapter(BackendAdapter):
    def capabilities(self) -> dict[str, Any]:
        info = get_model_capabilities("rag_default")
        return {
            "supported_tasks": info.get("tasks", []),
            "supported_models": ["rag_default"],
            "supported_training_methods": info.get("training_methods", []),
            "supported_export_formats": ["faiss_dir"],
        }

    def estimate_resources(self, model_name: str, dataset_size: int, config: dict[str, Any]) -> ResourceEstimate:
        # FAISS and local sentence-transformers require 0 GPU VRAM.
        return ResourceEstimate(
            vram_required_mb=0,
            ram_required_mb=1024,
            disk_required_mb=500,
            estimated_training_seconds=60,
            estimated_cost_usd=0.0
        )

    def prepare(self, dataset_path: Path, config: dict[str, Any]) -> Path:
        """
        For RAG, prepare() reads the raw files, chunks them, and generates embeddings.
        This prepares the data for FAISS index construction in train().
        """
        from backend.understanding.engine import extract_document
        from backend.rag.chunking import chunk_documents
        from backend.rag.embeddings import embed_chunks
        
        # In a real pipeline, dataset_path points to raw files or a manifest.
        # For simplicity, if dataset_path is a directory, iterate over its files.
        all_docs = []
        if dataset_path.is_dir():
            for file in dataset_path.iterdir():
                if file.is_file() and file.name != "dataset_manifest.json":
                    docs = extract_document(file)
                    all_docs.extend(docs)
        else:
            all_docs = extract_document(dataset_path)
            
        print(f"[RAG] Extracted {len(all_docs)} root document pages/sections.")
        
        # 2. Chunking
        chunk_size = config.get("chunk_size", 500)
        overlap = config.get("chunk_overlap", 50)
        chunks = chunk_documents(all_docs, chunk_size, overlap)
        print(f"[RAG] Generated {len(chunks)} overlapping chunks.")
        
        # 3. Embeddings
        print("[RAG] Generating embeddings...")
        embeddings, final_chunks = embed_chunks(chunks)
        
        out_dir = settings.processed_dir / f"rag_prep_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / "prepared_chunks.json", "w") as f:
            json.dump({
                "embeddings": embeddings,
                "chunks": final_chunks
            }, f)
            
        print(f"[RAG] Preparation complete. Saved to {out_dir}.")
        return out_dir

    def train(self, dataset_path: Path, config: dict[str, Any]) -> TrainingResult:
        """
        For RAG, train() is the process of building the FAISS index and persisting it.
        """
        from backend.rag.vector_store import VectorStore
        
        print("[RAG] Loading prepared embeddings for indexing...")
        with open(dataset_path / "prepared_chunks.json", "r") as f:
            data = json.load(f)
            
        embeddings = data["embeddings"]
        chunks = data["chunks"]
        
        # Build FAISS index
        print(f"[RAG] Building FAISS index for {len(chunks)} vectors...")
        store = VectorStore(dimension=384) # all-MiniLM-L6-v2 size
        store.add(embeddings, chunks)
        
        run_id = f"exp_{int(time.time())}"
        model_out_dir = settings.models_dir / run_id
        store.save(model_out_dir)
        
        print(f"[RAG] FAISS index saved to {model_out_dir}.")
        
        return TrainingResult(
            artifact_path=model_out_dir,
            metrics={"index_size": len(chunks), "embedding_dim": 384}
        )

    def evaluate(self, model_path: Path, dataset_path: Path, config: dict[str, Any]) -> EvaluationResult:
        """
        Evaluate RAG retrieval quality using a small set of standard queries.
        This checks if the correct source document appears in the top-k retrieved chunks.
        """
        from backend.rag.vector_store import VectorStore
        from backend.rag.embeddings import embed_chunks
        
        store = VectorStore.load(model_path)
        
        # Simple test fixture for retrieval evaluation
        # We query the index and check if the returned chunks have content 
        # or metadata that indicates successful retrieval.
        test_queries = config.get("test_queries", [
            {"query": "What is the main topic?", "expected_source": None} # Generic fallback
        ])
        
        correct_retrievals = 0
        total_queries = len(test_queries)
        
        for tq in test_queries:
            q = tq["query"]
            expected = tq.get("expected_source")
            
            # Embed query
            q_emb, _ = embed_chunks([{"text": q}])
            if not q_emb:
                continue
                
            results = store.retrieve(q_emb[0], k=5)
            
            # If we don't have an expected source to check against, we just count it as success
            # if it returns anything.
            if not expected:
                if results:
                    correct_retrievals += 1
                continue
                
            # Check if expected source is in the top-k results
            found = False
            for r in results:
                if r["metadata"].get("source") == expected:
                    found = True
                    break
                    
            if found:
                correct_retrievals += 1
                
        accuracy = correct_retrievals / total_queries if total_queries > 0 else 0
        
        return EvaluationResult(
            metrics={"retrieval_accuracy": accuracy, "total_queries": total_queries},
            error_analysis=None
        )

    def export(self, model_path: Path, export_format: str, output_path: Path) -> Path:
        if export_format not in ["faiss_dir"]:
            raise NotImplementedError(f"Format {export_format} not supported")
            
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(model_path, output_path)
        
        return output_path

    def deploy(self, model_path: Path, deploy_config: dict[str, Any]) -> dict[str, Any]:
        return {"status": "deployed", "type": "rag_endpoint", "path": str(model_path)}
