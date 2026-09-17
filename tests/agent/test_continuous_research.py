"""Research continues through repeated compaction and recoverable model failures."""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from papyrus_chat.agent.context import ResearchPolicy, ResearchRunState
from papyrus_chat.agent.context.compaction import bounded_history
from papyrus_chat.agent.context.evidence import EvidenceLedger
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.fixture()
def service(corpus_artifact: Path) -> CorpusService:
    return CorpusService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))


def test_oversized_notes_recover_and_survive_history_replay_and_compaction(service):
    initial = "Objective: investigate flax. Next: inspect candidates."
    replacement = (
        "Objective: flax at Aphrodito. Findings: λιν / ⲗⲓⲛ. Next: resolve dating. "
        "Unverified lead: https://papyri.info/ddbdp/invented"
    )
    answer = "Model-supplied background: research notes saved."
    deps = CorpusToolDeps(service)
    requests = 0

    def dialogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(
                [ToolCallPart("update_research_notes", json.dumps({"notes": initial}))]
            )
        if requests == 2:
            assert deps.research_state.ledger.notes == initial
            return ModelResponse(
                [ToolCallPart("update_research_notes", json.dumps({"notes": "λⲗ" * 2000 + "α"}))]
            )
        if requests == 3:
            assert deps.research_state.ledger.notes == initial
            retries = [part for part in messages[-1].parts if isinstance(part, RetryPromptPart)]
            assert len(retries) == 1
            errors = retries[0].content
            assert isinstance(errors, list)
            assert len(errors) == 1
            assert errors[0]["type"] == "string_too_long"
            assert tuple(errors[0]["loc"]) == ("notes",)
            assert "4000" in errors[0]["msg"]
            return ModelResponse(
                [ToolCallPart("update_research_notes", json.dumps({"notes": replacement}))]
            )
        if requests == 4:
            assert deps.research_state.ledger.notes == replacement
            return ModelResponse([ToolCallPart("get_research_progress", {})])
        assert requests == 5
        progress = next(part for part in messages[-1].parts if isinstance(part, ToolReturnPart))
        assert progress.tool_name == "get_research_progress"
        assert isinstance(progress.content, dict)
        assert progress.content["notes"] == replacement
        return ModelResponse([TextPart(answer)])

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        # Exercise retry/replay first; compact the restored ledger explicitly below.
        policy=ResearchPolicy(context_window=131072),
    )
    result = agent.run_sync("Investigate flax and save notes.", deps=deps)
    assert result.output == answer
    assert requests == 5

    messages = ModelMessagesTypeAdapter.validate_json(result.all_messages_json())
    replayed = EvidenceLedger()
    replayed.ingest(messages)
    for ledger in (deps.research_state.ledger, replayed):
        assert ledger.notes == replacement
        assert not ledger.records
        assert not ledger.corpus_urls
    assert not deps.known_corpus_urls

    state = ResearchRunState(ledger=replayed, question=deps.research_state.question)
    compacted = bounded_history(
        messages, state, ModelRequestParameters(), budget=8000, keep_recent=False
    )
    checkpoint_text = "\n".join(
        part.content
        for message in compacted
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )
    assert "Research notes (model-written)" in checkpoint_text
    assert replacement in checkpoint_text
    assert initial not in checkpoint_text


def test_default_research_continues_through_repeated_compaction(service):
    research = 0
    summaries = 0
    answer = "Inventory reviewed. Model-supplied interpretation: further textual work is possible."

    def dialogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal research, summaries
        if "papyrologist" not in (info.instructions or ""):
            summaries += 1
            return ModelResponse([TextPart("Inventory inspected; preserve original tool records.")])
        assert info.function_tools, "research tools must remain available"
        research += 1
        if research == 55:
            return ModelResponse([TextPart(answer)])
        return ModelResponse(
            [
                TextPart("Tentative interpretation πάπυρος " * 1_000),
                ToolCallPart("describe_corpus", {}, tool_call_id=f"inventory-{research}"),
            ]
        )

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=32768),
    )
    result = agent.run_sync("Research the inventory carefully.", deps=CorpusToolDeps(service))

    assert research == 55
    assert summaries > 3
    assert result.output.split("\n\nCoverage:")[0] == answer
    assert result.usage.requests == research + summaries


def test_failed_summary_uses_mechanical_compaction_and_keeps_researching(service):
    research = 0
    summaries = 0
    answer = "Model-supplied background: inventory examination is complete."

    def dialogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal research, summaries
        if "papyrologist" not in (info.instructions or ""):
            summaries += 1
            raise RuntimeError("summary provider temporarily unavailable")
        assert info.function_tools, "failed summaries must not disable research tools"
        research += 1
        if research == 5:
            assert "documents" in str(messages)
            return ModelResponse([TextPart(answer)])
        return ModelResponse(
            [
                TextPart("Tentative interpretation πάπυρος " * 1_000),
                ToolCallPart("describe_corpus", {}, tool_call_id=f"inventory-{research}"),
            ]
        )

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=32768),
    )
    result = agent.run_sync("Inspect inventory.", deps=CorpusToolDeps(service))

    assert summaries >= 1
    assert research == 5
    assert result.output.split("\n\nCoverage:")[0] == answer


def test_unset_generation_limit_is_not_sent_to_the_model(service):
    def dialogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert "max_tokens" not in (info.model_settings or {})
        return ModelResponse([TextPart("Model-supplied background: hello.")])

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(),
    )
    agent.run_sync("Hello.", deps=CorpusToolDeps(service))


def test_generation_exhaustion_recovers_once_and_counts_both_requests(service):
    requests = 0
    answer = "Model-supplied background: the complete answer."

    def dialogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert requests <= 2, "generation recovery must be bounded"
        if requests == 1:
            return ModelResponse(
                [TextPart("UNFINISHED_DRAFT")],
                finish_reason="length",
                usage=RequestUsage(input_tokens=100, output_tokens=200),
            )
        return ModelResponse(
            [TextPart(answer)], usage=RequestUsage(input_tokens=110, output_tokens=30)
        )

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=32768),
    )
    result = agent.run_sync("Give a complete answer.", deps=CorpusToolDeps(service))

    assert requests == 2
    assert result.output.split("\n\nCoverage:")[0] == answer
    assert result.usage.requests == 2
    assert result.usage.output_tokens == 230
