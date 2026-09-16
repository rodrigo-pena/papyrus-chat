"""Research decisions survive the runtime's mechanical compaction paths."""

from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.context.compaction import SUMMARY_INSTRUCTIONS
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.mark.parametrize("summary_skipped", [False, True])
def test_runtime_preserves_latest_decision_without_a_summary(
    corpus_artifact: Path, summary_skipped: bool
):
    service = CorpusService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))
    decision = "The inventory is sufficient; answer using the existing findings."
    latest = [
        ModelResponse(
            [TextPart(decision), ToolCallPart("describe_corpus", {}, "latest-inventory")]
        ),
        ModelRequest([ToolReturnPart("describe_corpus", {"documents": 10}, "latest-inventory")]),
    ]
    history = [
        ModelRequest([UserPromptPart("Describe the corpus inventory.")]),
        ModelResponse([TextPart("Earlier investigation. " * 5000)]),
        ModelRequest(parts=[]),
        *latest,
    ]
    summaries = 0
    requests = 0

    def dialogue(messages, info):
        nonlocal summaries, requests
        if SUMMARY_INSTRUCTIONS in (info.instructions or ""):
            summaries += 1
            assert not summary_skipped
            raise RuntimeError("Summary endpoint unavailable")
        requests += 1
        assert decision in str(messages)
        assert latest[0] in messages
        assert all(
            part in [retained for message in messages for retained in message.parts]
            for part in latest[1].parts
        )
        assert "Earlier investigation. " * 5000 not in str(messages)
        if summary_skipped:
            assert not info.function_tools
        return ModelResponse([TextPart("No corpus evidence was inspected.")])

    policy = ResearchPolicy(research_request_limit=1 if summary_skipped else None)
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=policy,
    )
    deps = CorpusToolDeps(service)
    result = agent.run_sync(message_history=history, deps=deps)
    assert result.output.startswith("No corpus evidence was inspected.")
    assert deps.research_state.compactions == 1
    assert summaries == (0 if summary_skipped else 1)
    assert requests == 1
