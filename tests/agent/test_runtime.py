"""Pydantic AI runtime configuration and citation validation."""

from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.test import TestModel

from papyrus_chat.agent.runtime import (
    RESEARCH_INSTRUCTIONS,
    create_research_agent,
    model_supports_native_web_search,
    validate_research_output,
)
from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.fixture()
def tool_service(tmp_path: Path, fixture_git_repo: Path) -> CorpusToolService:
    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )
    return CorpusToolService(StructuredCorpusSearch(artifact / "corpus.sqlite"))


def test_runtime_uses_existing_openai_compatible_configuration(
    tool_service: CorpusToolService,
) -> None:
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        tool_service,
        model=TestModel(),
    )

    assert RESEARCH_INSTRUCTIONS
    assert isinstance(agent, Agent)


def test_runtime_registers_tools_without_a_real_model_call(
    tool_service: CorpusToolService,
) -> None:
    model = TestModel(custom_output_text="Model-supplied background: the corpus is local.")
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        tool_service,
        model=model,
    )

    agent.run_sync("Describe the corpus.", deps=CorpusToolDeps(service=tool_service))

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert {tool.name for tool in parameters.function_tools} == {
        "describe_corpus",
        "search_documents",
        "inspect_documents",
        "facet_documents",
    }


def test_native_web_search_is_opted_in_only_for_responses_models() -> None:
    assert model_supports_native_web_search("openai-responses:gpt-5.2")
    assert not model_supports_native_web_search("gpt-5.2")


def test_output_validator_accepts_known_corpus_links_and_rejects_unknown_links() -> None:
    known = {"https://papyri.info/ddbdp/p.mich;8;480"}
    answer = "Corpus evidence: [1](https://papyri.info/ddbdp/p.mich;8;480)."

    assert validate_research_output(answer, known) == answer

    with pytest.raises(ModelRetry, match="known corpus citation"):
        validate_research_output("Corpus evidence: [1](https://papyri.info/ddbdp/unknown).", known)


def test_output_validator_allows_memory_background_without_corpus_citation() -> None:
    answer = "Model-supplied background: Egyptian month names varied by period."

    assert validate_research_output(answer, set()) == answer


def test_output_validator_allows_a_clear_no_evidence_answer() -> None:
    answer = "Scope and method: the displayed filters returned no corpus evidence."

    assert validate_research_output(answer, set()) == answer
