"""Memory tool contracts are tested through model-facing argument validation."""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from papyrus_chat.agent.context.evidence import EvidenceLedger
from papyrus_chat.agent.context.memory import ResearchNotes, register_memory_tools


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"notes": "λⲗ" * 2000 + "α"}, "string_too_long"),
        ({}, "missing"),
        ({"notes": 42}, "string_type"),
        ({"notes": {"notes": "summary"}}, "string_type"),
    ],
)
def test_invalid_notes_report_field_error_and_preserve_saved_notes(serialized, payload, error_type):
    ledger = EvidenceLedger(notes="Previous findings")
    deps = SimpleNamespace(research_state=SimpleNamespace(ledger=ledger))

    def dialogue(messages, info):
        if len(messages) == 1:
            args = json.dumps(payload) if serialized else payload
            return ModelResponse([ToolCallPart("update_research_notes", args)])
        retries = [part for part in messages[-1].parts if isinstance(part, RetryPromptPart)]
        assert len(retries) == 1
        errors = retries[0].content
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert errors[0]["type"] == error_type
        assert tuple(errors[0]["loc"]) == ("notes",)
        if error_type == "string_too_long":
            assert "4000" in errors[0]["msg"]
        assert ledger.notes == "Previous findings"
        return ModelResponse([TextPart("Done")])

    agent = Agent(FunctionModel(dialogue))
    register_memory_tools(agent)
    agent.run_sync("Save notes", deps=deps)
    assert ledger.notes == "Previous findings"


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("notes", ["", "λⲗ" * 2000])
def test_valid_notes_preserve_text_and_advertise_bounded_string(serialized, notes):
    ledger = EvidenceLedger(notes="Previous findings")
    deps = SimpleNamespace(research_state=SimpleNamespace(ledger=ledger))

    def dialogue(messages, info):
        tool = next(tool for tool in info.function_tools if tool.name == "update_research_notes")
        schema = tool.parameters_json_schema
        assert schema["type"] == "object"
        assert schema["required"] == ["notes"]
        assert set(schema["properties"]) == {"notes"}
        assert schema["properties"]["notes"]["type"] == "string"
        assert schema["properties"]["notes"]["maxLength"] == 4000
        if len(messages) == 1:
            payload = {"notes": notes}
            args = json.dumps(payload) if serialized else payload
            return ModelResponse([ToolCallPart("update_research_notes", args)])
        assert ledger.notes == notes
        assert messages[-1].parts[0].content == ResearchNotes(notes=notes)
        return ModelResponse([TextPart("Done")])

    agent = Agent(FunctionModel(dialogue))
    register_memory_tools(agent)
    agent.run_sync("Save notes", deps=deps)
    assert ledger.notes == notes
