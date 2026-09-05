"""Groq API client — BIBLE §40, ARCHITECTURE.md §4 (Locked Decision).

Thin wrapper around the Groq SDK configured for GPT OSS 120B
tool-calling (Llama 3.3 70B deprecated). Reads GROQ_API_KEY from env via config.py.

All orchestrator tool definitions (ARCHITECTURE.md §4) are registered
here and sent with every chat-completion call.
"""

from __future__ import annotations

import json
from typing import Any

from groq import Groq

from backend.config import settings
from backend.orchestrator.tools import TOOL_DEFINITIONS


import logging

logger = logging.getLogger(__name__)

_configured_keys: list[str] = []
_active_key_index: int = 0

def _init_keys() -> None:
    global _configured_keys
    if not _configured_keys:
        keys = [
            settings.groq_api_key_1,
            settings.groq_api_key_2,
            settings.groq_api_key_3,
            settings.groq_api_key_4,
            settings.groq_api_key_5,
        ]
        _configured_keys = [k for k in keys if k]
        if not _configured_keys:
            raise RuntimeError(
                "GROQ_API_KEY_1 is not set.  Copy backend/.env.example to "
                "backend/.env and fill in your key."
            )

def _get_active_client() -> Groq:
    _init_keys()
    return Groq(api_key=_configured_keys[_active_key_index])

def _rotate_key() -> None:
    global _active_key_index
    _init_keys()
    old_index = _active_key_index
    _active_key_index = (_active_key_index + 1) % len(_configured_keys)
    logger.warning(f"Groq API Rate Limit reached. Rotating key from index {old_index} to {_active_key_index}.")


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Send a chat-completion request to GPT OSS 120B via Groq.

    Parameters
    ----------
    messages : list[dict]
        OpenAI-format message list (role + content).
    tools : list[dict] | None
        Override tool definitions.  Defaults to the canonical set from
        ARCHITECTURE.md §4.
    max_tokens : int
        Max response tokens.

    Returns
    -------
    dict with keys:
        - "content": str | None           (text reply)
        - "tool_calls": list[dict] | None (tool-call requests)
        - "raw": the full Groq response object
    """
    _init_keys()
    
    kwargs: dict[str, Any] = {
        "model": settings.groq_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resolved_tools = tools if tools is not None else TOOL_DEFINITIONS
    if resolved_tools:
        kwargs["tools"] = resolved_tools
        kwargs["tool_choice"] = "auto"

    import groq
    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        
        # Try all available keys before doing exponential backoff
        for key_attempt in range(len(_configured_keys)):
            client = _get_active_client()
            try:
                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message
            
                tool_calls = None
                if message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        }
                        for tc in message.tool_calls
                    ]
            
                return {
                    "content": message.content,
                    "tool_calls": tool_calls,
                    "raw": response,
                }
            except groq.RateLimitError as e:
                if key_attempt < len(_configured_keys) - 1:
                    _rotate_key()
                    continue
                else:
                    rate_limit_err = e
                    break
            except groq.BadRequestError as e:
                if "tool_use_failed" in str(e) and attempt < max_retries - 1:
                    # Add a system reminder about JSON formatting
                    if not any(m.get("content") == "System: Fix your tool format." for m in kwargs["messages"]):
                        kwargs["messages"].append({
                            "role": "user",
                            "content": "System: Fix your tool format. Please output valid JSON only without any markdown blocks, newlines, or extra tags inside the tool payload."
                        })
                    bad_req_err = e
                    break
                raise
                
        if 'rate_limit_err' in locals() and attempt < max_retries - 1:
            sleep_time = 15 * (attempt + 1)
            logger.warning(f"All keys rate limited. Sleeping for {sleep_time}s to clear RPM window...")
            time.sleep(sleep_time)
            del rate_limit_err
            continue
            
        if 'bad_req_err' in locals() and attempt < max_retries - 1:
            del bad_req_err
            continue
            
        if 'rate_limit_err' in locals():
            raise rate_limit_err
            
    raise RuntimeError("Exhausted retries in Groq client")


def run_orchestrator_loop(project_id: str, goal: str, expert_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute the Phase 3 planning loop using tool calling and validation gate."""
    from backend.orchestrator.validation_gate import validate_and_execute
    from backend.orchestrator.tools import TOOL_DEFINITIONS, STAGE_TOOLS, TRANSITION_TOOL
    
    current_stage = "FORMULATION"
    
    system_prompt = (
        "You are the AI Orchestrator for the Unified AI Platform.\n"
        "Your task is to plan a machine learning pipeline for the user.\n"
        "You operate in stages: DISCOVERY -> FORMULATION -> DATASET -> CAPABILITY -> EXECUTION -> EVALUATION -> DEPLOYMENT.\n"
        "You currently only have access to tools for your ACTIVE STAGE. To get different tools, you must call `transition_stage` to move to the appropriate stage.\n"
        "You MUST first transition to the DATASET stage and call `create_dataset` to generate a dataset_version_id.\n"
        "Once you have the dataset_version_id, you MUST transition to EXECUTION and call `create_experiment`.\n"
        "For tabular data predicting a target, use model_name='autogluon_tabular', "
        "backend='autogluon', training_method='ensemble'.\n"
        "For text generation/chat, use backend='unsloth' and pick the most appropriate model_name from the capability registry (e.g. 'unsloth_qwen2.5_3b', 'unsloth_llama3.2_1b', etc.).\n"
        "For document Q&A or RAG (Retrieval-Augmented Generation), use model_name='rag_default', backend='rag', training_method='faiss_index'.\n"
        "DO NOT ask the user for missing information. DO NOT hallucinate tools like 'get_dataset_versions'.\n"
        "Do NOT call `start_training`."
    )
    
    if expert_config:
        system_prompt += f"\nEXPERT CONSTRAINTS: The user has provided expert configurations. When you call `create_experiment`, you MUST pass the following JSON exactly into the `config_json` parameter: {json.dumps(expert_config)}"
    
    # Pass the understanding report in the user message so it doesn't need to call inspect_files
    from backend.understanding.engine import analyze_project
    report = analyze_project(project_id)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Active Stage: {current_stage}\nProject ID: {project_id}\nGoal: {goal}\nData Report: {json.dumps(report)}"}
    ]
    
    formulate_tool = {
        "type": "function",
        "function": {
            "name": "formulate_problem",
            "description": "Formulate a problem spec from a goal and understanding report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "understanding_report": {"type": "object"},
                },
                "required": ["goal", "understanding_report"],
            },
        },
    }
    
    all_tools = list(TOOL_DEFINITIONS) + [formulate_tool, TRANSITION_TOOL]
    
    def get_stage_tools(stage: str) -> list[dict]:
        allowed_names = STAGE_TOOLS.get(stage, [])
        return [t for t in all_tools if t["function"]["name"] in allowed_names] + [TRANSITION_TOOL]
    
    max_turns = 10
    
    for _ in range(max_turns):
        stage_tools = get_stage_tools(current_stage)
        try:
            response = chat(messages, tools=stage_tools)
        except RuntimeError:
            # Zero-config / offline / test fallback pipeline orchestration
            from backend.orchestrator.problem_formulator import formulate_problem
            from backend.db import SessionLocal, DataSource
            spec = formulate_problem(goal, report)
            
            db = SessionLocal()
            try:
                datasources = db.query(DataSource).filter_by(project_id=project_id).all()
                ds_ids = ",".join(d.id for d in datasources) if datasources else "mock_ds"
            finally:
                db.close()
                
            task_type = spec.get("task_type", "classification") if isinstance(spec, dict) else "classification"
            ds_res = validate_and_execute("create_dataset", {
                "project_id": project_id,
                "datasource_ids": ds_ids,
                "task_type": task_type
            })
            ds_version_id = ds_res.get("dataset_version_id", "v1") if isinstance(ds_res, dict) else "v1"
            
            if task_type == "rag":
                model_name = "rag_default"
                backend = "rag"
                training_method = "faiss_index"
                config = {}
            elif isinstance(spec, dict) and spec.get("modality") == "tabular":
                model_name = "autogluon_best"
                backend = "autogluon"
                training_method = "ensemble"
                config = {"target_column": spec.get("target_column") or "target"}
            else:
                model_name = "unsloth_llama3.2_3b"
                backend = "unsloth"
                training_method = "lora"
                config = {}
                
            if expert_config:
                config.update(expert_config)
                
            exp_res = validate_and_execute("create_experiment", {
                "project_id": project_id,
                "dataset_version_id": ds_version_id,
                "model_name": model_name,
                "backend": backend,
                "training_method": training_method,
                "config_json": config
            })
            return {"status": "success", "experiment": exp_res, "transcript": messages}
        message = response["raw"].choices[0].message
        
        # Append assistant's response to history
        messages.append(message.model_dump(exclude_unset=True))
        
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            print(f"\n[GPT-OSS-120B BEHAVIOR] Model returned text instead of a tool call: {response.get('content')}")
            # Reached a final answer without further tool calls
            return {"status": "finished", "final_answer": response.get("content")}
            
        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            
            valid_tool_names = [t["function"]["name"] for t in stage_tools]
            
            if name not in valid_tool_names:
                result = {"error": f"Tool '{name}' is not available in the '{current_stage}' stage. Use 'transition_stage' to move to the appropriate stage (e.g. DATASET, EXECUTION)."}
            elif name == "transition_stage":
                next_stage = args.get("next_stage")
                if next_stage in STAGE_TOOLS:
                    current_stage = next_stage
                    result = {"status": "success", "message": f"Successfully transitioned to {current_stage}. You now have access to {current_stage} tools."}
                else:
                    result = {"error": f"Invalid stage '{next_stage}'."}
            elif name == "formulate_problem":
                from backend.orchestrator.problem_formulator import formulate_problem
                result = formulate_problem(args.get("goal", goal), args.get("understanding_report", {}))
            else:
                result = validate_and_execute(name, args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result)
            })
            
            if name == "create_experiment" and "error" not in result:
                return {"status": "success", "experiment": result, "transcript": messages}
                
    return {"status": "error", "detail": "Exceeded maximum turns without creating an experiment."}
