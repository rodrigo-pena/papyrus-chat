"""Regression coverage for agents that keep researching instead of answering."""

from pathlib import Path

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.fixture()
def service(corpus_artifact: Path) -> CorpusToolService:
    return CorpusToolService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))


class EndlessResearch:
    """Keep requesting real corpus tools until the runtime removes them."""

    def __init__(self, *, invalid_final: bool = False) -> None:
        self.research_requests = 0
        self.final_requests = 0
        self.invalid_final = invalid_final

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        if info.function_tools:
            self.research_requests += 1
            assert self.research_requests <= 8, "runtime did not stop research"
            return ModelResponse(
                [
                    ToolCallPart(
                        "describe_corpus", {}, tool_call_id=f"describe-{self.research_requests}"
                    )
                ]
            )
        self.final_requests += 1
        assert not info.model_request_parameters.native_tools
        text = (
            "Corpus evidence: https://papyri.info/ddbdp/invented;1;999"
            if self.invalid_final
            else "Research is incomplete. No corpus evidence was inspected."
        )
        return ModelResponse([TextPart(text)])


def test_endless_research_transitions_to_a_tool_free_answer(service: CorpusToolService) -> None:
    dialogue = EndlessResearch()
    policy = ResearchPolicy(context_window=131_072, research_request_limit=3, compaction_limit=0)
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=policy,
        enable_web_search=True,
    )

    result = agent.run_sync("Investigate the corpus thoroughly.", deps=CorpusToolDeps(service))

    assert dialogue.research_requests == 3
    assert dialogue.final_requests == 1
    assert "incomplete" in result.output
    assert result.usage.requests == 4


def test_finalization_citation_repair_cannot_restart_research(service: CorpusToolService) -> None:
    dialogue = EndlessResearch(invalid_final=True)
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=131_072, research_request_limit=2, compaction_limit=0),
    )

    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync("Investigate the corpus.", deps=CorpusToolDeps(service))

    assert dialogue.research_requests == 2
    assert dialogue.final_requests == 2


def test_oversized_current_question_is_rejected_before_request(service: CorpusToolService) -> None:
    dialogue = EndlessResearch()
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=32_768),
    )

    with pytest.raises(Exception, match="(?i)(prompt|question).*(large|fit|context|exceed)"):
        agent.run_sync("πάπυρος " * 100_000, deps=CorpusToolDeps(service))

    assert dialogue.research_requests == 0
    assert dialogue.final_requests == 0


def test_long_tool_run_compacts_and_still_delivers_an_answer(service: CorpusToolService) -> None:
    research_requests = 0
    summary_requests = 0
    final_requests = 0
    final_history = ""

    def dialogue(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal research_requests, summary_requests, final_requests, final_history
        if info.function_tools:
            research_requests += 1
            assert research_requests <= 6, "runtime did not bound research"
            return ModelResponse(
                [
                    TextPart("Tentative interpretation: πάπυρος. " * 1_000),
                    ToolCallPart("describe_corpus", {}, tool_call_id=f"long-{research_requests}"),
                ]
            )
        if "papyrologist" not in (info.instructions or ""):
            summary_requests += 1
            assert summary_requests <= 2
            return ModelResponse(
                [TextPart("The user requested corpus research. Only inventory was inspected.")]
            )
        final_requests += 1
        final_history = str(messages)
        return ModelResponse(
            [TextPart("Research is incomplete. No corpus evidence was inspected.")]
        )

    policy = ResearchPolicy(context_window=32_768, research_request_limit=6, compaction_limit=2)
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=policy,
    )

    result = agent.run_sync(
        "Investigate corpus inventory and uncertainties.", deps=CorpusToolDeps(service)
    )

    assert 1 <= summary_requests <= 2
    assert research_requests + summary_requests <= 6
    assert final_requests == 1
    assert "documents" in final_history
    assert "Investigate corpus inventory and uncertainties." in final_history
    assert "incomplete" in result.output


def test_finalization_removes_native_web_tools(service: CorpusToolService) -> None:
    from pydantic_ai import WebSearchTool
    from pydantic_ai.capabilities import NativeTool

    calls = []

    def dialogue(messages, info):
        calls.append(info.model_request_parameters)
        if info.function_tools:
            return ModelResponse([ToolCallPart("describe_corpus", {}, "inventory")])
        return ModelResponse(
            [TextPart("Research is incomplete. No corpus evidence was inspected.")]
        )

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=131072, research_request_limit=1),
    )
    agent.run_sync(
        "Investigate", deps=CorpusToolDeps(service), capabilities=[NativeTool(WebSearchTool())]
    )
    assert calls[0].native_tools
    assert not calls[-1].native_tools
    assert not calls[-1].function_tools


def test_failed_summary_transitions_to_final_answer(service: CorpusToolService) -> None:
    counts = {"research": 0, "summary": 0, "final": 0}

    def dialogue(messages, info):
        if info.function_tools:
            counts["research"] += 1
            return ModelResponse(
                [TextPart("πάπυρος " * 10000), ToolCallPart("describe_corpus", {}, "inventory")]
            )
        if "papyrologist" not in (info.instructions or ""):
            counts["summary"] += 1
            raise RuntimeError("Endpoint unavailable")
        counts["final"] += 1
        return ModelResponse(
            [TextPart("Research is incomplete. No corpus evidence was inspected.")]
        )

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(),
    )
    result = agent.run_sync("Investigate", deps=CorpusToolDeps(service))
    assert counts == {"research": 1, "summary": 1, "final": 1}
    assert "incomplete" in result.output


def test_followup_resets_budget_even_with_reused_python_dependencies(service: CorpusToolService):
    dialogue = EndlessResearch()
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=131072, research_request_limit=1),
    )
    deps = CorpusToolDeps(service)
    first = agent.run_sync("Investigate", deps=deps)
    agent.run_sync("Continue", deps=deps, message_history=first.all_messages())
    assert dialogue.research_requests == 2
    assert dialogue.final_requests == 2
