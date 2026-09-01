"""Deterministic research-dialogue evaluations with no real model calls."""

from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.fixture()
def tool_service(tmp_path, fixture_git_repo) -> CorpusToolService:
    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )
    return CorpusToolService(StructuredCorpusSearch(artifact / "corpus.sqlite"))


def _returned_tool_names(messages: list[ModelMessage]) -> list[str]:
    return [
        part.tool_name
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


class ResearchDialogue:
    """A small scripted model that checks the agent's intended tool sequence."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        returned = _returned_tool_names(messages)
        if not returned:
            self.calls.append(("describe_corpus", {}))
            return ModelResponse([ToolCallPart("describe_corpus", {}, tool_call_id="describe-1")])
        if returned == ["describe_corpus"]:
            arguments = {
                "term_groups": [["Κλαύδιος", "Claudius"], ["Geld", "δραχμή"]],
                "fields": ["transcription", "metadata"],
                "transcription_languages": ["grc"],
                "date_interval": {"not_before": 101, "not_after": 125},
                "limit": 10,
            }
            self.calls.append(("search_documents", arguments))
            return ModelResponse(
                [ToolCallPart("search_documents", arguments, tool_call_id="search-1")]
            )
        if returned == ["describe_corpus", "search_documents"]:
            arguments = {"document_ids": ["ddbdp:DDbDP/27/27093.xml"], "excerpt_limit": 1}
            self.calls.append(("inspect_documents", arguments))
            return ModelResponse(
                [ToolCallPart("inspect_documents", arguments, tool_call_id="inspect-1")]
            )
        assert returned == ["describe_corpus", "search_documents", "inspect_documents"]
        return ModelResponse(
            [
                TextPart(
                    "Scope and method: ddbdp, Greek transcription, and German metadata "
                    "terms Κλαύδιος/Claudius plus Geld/δραχμή were searched for 101-125. "
                    "Corpus evidence: https://papyri.info/ddbdp/p.mich;8;480. "
                    "Transcription evidence is separated from model-supplied background."
                )
            ]
        )


def test_research_dialogue_uses_bounded_multistep_corpus_tools(
    tool_service: CorpusToolService,
) -> None:
    script = ResearchDialogue()
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        tool_service,
        model=FunctionModel(script),
    )
    deps = CorpusToolDeps(service=tool_service)

    result = agent.run_sync("Find documentary evidence and explain its date.", deps=deps)

    assert result.output.startswith("Scope and method:")
    assert [name for name, _ in script.calls] == [
        "describe_corpus",
        "search_documents",
        "inspect_documents",
    ]
    assert script.calls[1][1]["term_groups"] == [
        ["Κλαύδιος", "Claudius"],
        ["Geld", "δραχμή"],
    ]
    assert script.calls[1][1]["date_interval"] == {"not_before": 101, "not_after": 125}
    assert deps.known_corpus_urls == {"https://papyri.info/ddbdp/p.mich;8;480"}


class RefinementDialogue:
    """Script refinement to ensure the second run receives the first run history."""

    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []
        self.seen_history_on_refinement = False

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        user_prompts = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        is_refinement = bool(user_prompts) and "narrow" in str(user_prompts[-1]).casefold()
        last_message_is_new_prompt = isinstance(messages[-1], ModelRequest) and any(
            isinstance(part, UserPromptPart) for part in messages[-1].parts
        )
        if is_refinement and self.searches and last_message_is_new_prompt:
            self.seen_history_on_refinement = any(
                isinstance(part, ToolReturnPart) for message in messages for part in message.parts
            )
            arguments = {
                "term_groups": [["Κλαύδιος", "Claudius"], ["Geld"]],
                "fields": ["transcription", "metadata"],
                "date_interval": {"not_before": 101, "not_after": 125},
                "limit": 5,
            }
            self.searches.append(arguments)
            return ModelResponse(
                [ToolCallPart("search_documents", arguments, tool_call_id="refine-search")]
            )
        if not self.searches:
            arguments = {
                "term_groups": [["Κλαύδιος"]],
                "fields": ["transcription"],
                "limit": 20,
            }
            self.searches.append(arguments)
            return ModelResponse(
                [ToolCallPart("search_documents", arguments, tool_call_id="initial-search")]
            )
        return ModelResponse(
            [
                TextPart(
                    "Scope and method: the refined corpus evidence is https://papyri.info/ddbdp/p.mich;8;480."
                )
            ]
        )


def test_followup_run_can_refine_a_prior_query(
    tool_service: CorpusToolService,
) -> None:
    script = RefinementDialogue()
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        tool_service,
        model=FunctionModel(script),
    )
    deps = CorpusToolDeps(service=tool_service)

    first = agent.run_sync("Find Greek evidence about Claudius.", deps=deps)
    second = agent.run_sync(
        "Narrow this prior search to the linked date and German metadata.",
        deps=deps,
        message_history=first.all_messages(),
    )

    assert first.output.startswith("Scope and method:")
    assert second.output.startswith("Scope and method:")
    assert script.seen_history_on_refinement
    assert script.searches[0]["term_groups"] == [["Κλαύδιος"]]
    assert script.searches[1]["date_interval"] == {"not_before": 101, "not_after": 125}


class InsufficientEvidenceDialogue:
    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if not _returned_tool_names(messages):
            return ModelResponse(
                [
                    ToolCallPart(
                        "search_documents",
                        {"term_groups": [["not-present"]], "fields": ["transcription"]},
                        tool_call_id="empty-search",
                    )
                ]
            )
        return ModelResponse(
            [
                TextPart(
                    "Scope and method: the displayed filters returned no corpus evidence. "
                    "Model-supplied background is not a substitute for a corpus citation."
                )
            ]
        )


def _search_retry_count(messages: list[ModelMessage]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart) and part.tool_name == "search_documents"
    )


class RecoveringSearchDialogue:
    """Two invalid searches, then a valid one; the default retry budget of 1 aborted this run."""

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        retries = _search_retry_count(messages)
        if "search_documents" not in _returned_tool_names(messages):
            if retries < 2:
                return ModelResponse(
                    [
                        ToolCallPart(
                            "search_documents",
                            {
                                "term_groups": [["Κλαύδιος"]],
                                "fields": ["transcription"],
                                "date_interval": {"not_before": 125, "not_after": 101},
                            },
                            tool_call_id=f"invalid-search-{retries + 1}",
                        )
                    ]
                )
            return ModelResponse(
                [
                    ToolCallPart(
                        "search_documents",
                        {
                            "term_groups": [["Κλαύδιος", "Claudius"], ["Geld", "δραχμή"]],
                            "fields": ["transcription", "metadata"],
                            "transcription_languages": ["grc"],
                            "date_interval": {"not_before": 101, "not_after": 125},
                            "limit": 10,
                        },
                        tool_call_id="recovered-search",
                    )
                ]
            )
        return ModelResponse(
            [
                TextPart(
                    "Scope and method: ddbdp, Greek transcription, and German metadata "
                    "terms Κλαύδιος/Claudius plus Geld/δραχμή were searched for 101-125. "
                    "Corpus evidence: https://papyri.info/ddbdp/p.mich;8;480. "
                    "Transcription evidence is separated from model-supplied background."
                )
            ]
        )


def test_invalid_search_arguments_are_corrected_within_the_retry_budget(
    tool_service: CorpusToolService,
) -> None:
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        tool_service,
        model=FunctionModel(RecoveringSearchDialogue()),
    )
    deps = CorpusToolDeps(service=tool_service)

    result = agent.run_sync("Find documentary evidence and explain its date.", deps=deps)

    retry_parts = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart) and part.tool_name == "search_documents"
    ]
    assert len(retry_parts) == 2
    assert "not_before must be less than or equal" in str(retry_parts[0].content)
    assert result.output.startswith("Scope and method:")
    assert deps.known_corpus_urls == {"https://papyri.info/ddbdp/p.mich;8;480"}


def test_dialogue_labels_insufficient_evidence_and_background(
    tool_service: CorpusToolService,
) -> None:
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        tool_service,
        model=FunctionModel(InsufficientEvidenceDialogue()),
    )

    result = agent.run_sync(
        "Search for a term that is absent and explain what can be concluded.",
        deps=CorpusToolDeps(service=tool_service),
    )

    assert "no corpus evidence" in result.output
    assert "Model-supplied background" in result.output
