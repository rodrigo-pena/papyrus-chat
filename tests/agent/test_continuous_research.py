"""Research continues through repeated compaction and recoverable model failures."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.fixture()
def service(corpus_artifact: Path) -> CorpusToolService:
    return CorpusToolService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))


def test_default_research_continues_beyond_old_request_and_compaction_limits(service):
    research = 0
    summaries = 0
    answer = "Inventory reviewed. Model-supplied interpretation: further textual work is possible."

    def dialogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal research, summaries
        if "papyrologist" not in (info.instructions or ""):
            summaries += 1
            return ModelResponse([TextPart("Inventory inspected; preserve original tool records.")])
        assert info.function_tools, "default research must not enter mandatory finalization"
        research += 1
        if research == 21:
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

    assert research == 21
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
