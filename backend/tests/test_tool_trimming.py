"""Test stage-scoped tool trimming without calling Groq."""
import json
import pytest
from unittest.mock import patch, MagicMock

import backend.orchestrator.groq_client as groq_client
from backend.orchestrator.groq_client import run_orchestrator_loop
from backend.orchestrator.tools import STAGE_TOOLS, TOOL_DEFINITIONS, TRANSITION_TOOL

@patch("backend.orchestrator.groq_client.chat")
def test_stage_scoped_tool_trimming(mock_chat):
    # Turn 1: In FORMULATION, call transition_stage to DATASET
    resp_1 = MagicMock()
    resp_1.choices[0].message.tool_calls = [
        MagicMock(id="1", function=MagicMock(name="transition_stage", arguments=json.dumps({"next_stage": "DATASET"})))
    ]
    resp_1.choices[0].message.content = None
    
    # Turn 2: In DATASET, call create_dataset
    resp_2 = MagicMock()
    resp_2.choices[0].message.tool_calls = [
        MagicMock(id="2", function=MagicMock(name="create_dataset", arguments=json.dumps({})))
    ]
    resp_2.choices[0].message.content = None
    
    # Turn 3: In DATASET, call transition_stage to EXECUTION
    resp_3 = MagicMock()
    resp_3.choices[0].message.tool_calls = [
        MagicMock(id="3", function=MagicMock(name="transition_stage", arguments=json.dumps({"next_stage": "EXECUTION"})))
    ]
    resp_3.choices[0].message.content = None
    
    # Turn 4: In EXECUTION, call create_experiment (terminates)
    resp_4 = MagicMock()
    resp_4.choices[0].message.tool_calls = [
        MagicMock(id="4", function=MagicMock(name="create_experiment", arguments=json.dumps({})))
    ]
    resp_4.choices[0].message.content = None

    mock_chat.side_effect = [
        {"raw": resp_1, "tool_calls": [{"id": "1", "name": "transition_stage", "arguments": {"next_stage": "DATASET"}}], "content": None},
        {"raw": resp_2, "tool_calls": [{"id": "2", "name": "create_dataset", "arguments": {}}], "content": None},
        {"raw": resp_3, "tool_calls": [{"id": "3", "name": "transition_stage", "arguments": {"next_stage": "EXECUTION"}}], "content": None},
        {"raw": resp_4, "tool_calls": [{"id": "4", "name": "create_experiment", "arguments": {}}], "content": None}
    ]

    with patch("backend.orchestrator.validation_gate.validate_and_execute", return_value={"status": "mocked"}), \
         patch("backend.orchestrator.problem_formulator.formulate_problem", return_value={"status": "mocked"}), \
         patch("backend.understanding.engine.analyze_project", return_value={"mock": "report"}):
        res = run_orchestrator_loop("proj1", "predict stuff")
        
    assert res["status"] == "success", res
    assert mock_chat.call_count == 4
    
    calls = mock_chat.call_args_list
    
    # Turn 1: FORMULATION
    tools_turn_1 = calls[0].kwargs.get("tools", [])
    names_1 = [t["function"]["name"] for t in tools_turn_1]
    assert set(names_1) == set(STAGE_TOOLS["FORMULATION"] + ["transition_stage"])
    
    # Turn 2: DATASET
    tools_turn_2 = calls[1].kwargs.get("tools", [])
    names_2 = [t["function"]["name"] for t in tools_turn_2]
    assert set(names_2) == set(STAGE_TOOLS["DATASET"] + ["transition_stage"])
    
    # Turn 3: DATASET
    tools_turn_3 = calls[2].kwargs.get("tools", [])
    names_3 = [t["function"]["name"] for t in tools_turn_3]
    assert set(names_3) == set(STAGE_TOOLS["DATASET"] + ["transition_stage"])
    
    # Turn 4: EXECUTION
    tools_turn_4 = calls[3].kwargs.get("tools", [])
    names_4 = [t["function"]["name"] for t in tools_turn_4]
    assert set(names_4) == set(STAGE_TOOLS["EXECUTION"] + ["transition_stage"])
    
    # Print schema sizes
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
    
    full_tools_json = json.dumps(all_tools)
    trimmed_tools_json = json.dumps(tools_turn_4)
    print(f"\n--- TOKEN REDUCTION REPORT ---")
    print(f"Full toolset string size: {len(full_tools_json)} characters")
    print(f"Trimmed toolset (EXECUTION) string size: {len(trimmed_tools_json)} characters")
    print(f"Reduction: ~{100 - (len(trimmed_tools_json)/len(full_tools_json)*100):.1f}% smaller")
