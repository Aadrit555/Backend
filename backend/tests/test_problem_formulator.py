"""Tests for the Problem Formulator — Phase 3."""

from __future__ import annotations

from backend.orchestrator.problem_formulator import formulate_problem


def test_formulate_problem_rag() -> None:
    """1. answers questions about my company's PDFs + PDF report -> RAG"""
    goal = "answers questions about my company's PDFs"
    report = {
        "project_id": "proj_1",
        "file_type_counts": {"pdf": 10},
        "sources": [
            {"filename": "doc1.pdf", "type": "text", "looks_tabular": False, "word_count_estimate": 5000}
        ]
    }
    
    spec = formulate_problem(goal, report)
    
    assert spec["needs_training"] is False
    assert spec["task_type"] == "rag"
    assert spec["modality"] == "text"
    assert "rationale" in spec


def test_formulate_problem_fine_tuning() -> None:
    """2. consistently respond in my personal writing style + chat log -> Fine Tuning"""
    goal = "consistently respond in my personal writing style"
    report = {
        "project_id": "proj_2",
        "file_type_counts": {"json": 1},
        "sources": [
            {"filename": "chat_logs.json", "type": "json", "structure": "list", "inferred_schema": {"message": "str", "sender": "str"}}
        ]
    }
    
    spec = formulate_problem(goal, report)
    
    assert spec["needs_training"] is True
    assert spec["task_type"] == "fine_tuning"
    assert spec["modality"] == "text"
    assert "rationale" in spec


def test_formulate_problem_tabular_classification() -> None:
    """3. tabular understanding report + predict machine failure -> Classification"""
    goal = "predict machine failure"
    report = {
        "project_id": "proj_3",
        "file_type_counts": {"csv": 1},
        "sources": [
            {
                "filename": "sensor_data.csv",
                "type": "tabular",
                "row_count": 1000,
                "likely_target_column": "failed",
                "columns": [
                    {"name": "failed", "dtype": "bool", "null_count": 0, "cardinality": 2}
                ]
            }
        ]
    }
    spec = formulate_problem(goal, report)
    
    assert spec["needs_training"] is True
    assert spec["task_type"] == "classification"
    assert spec["modality"] == "tabular"
    # The LLM should reasonably infer the target_column from the report's likely_target_column
    # We won't strictly assert the exact string to avoid flakiness.
    # Qwen might return None, so we just assert task_type and modality.
    assert "rationale" in spec
