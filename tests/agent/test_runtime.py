"""Pydantic AI runtime configuration and citation validation."""

from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.test import TestModel

import papyrus_chat.agent.runtime as runtime
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
from papyrus_chat.retrieval.structured import CorpusDocumentMatch, StructuredCorpusSearch


def _match(url: str, mapping: dict[str, tuple[str, str, str]]) -> CorpusDocumentMatch | None:
    record = mapping.get(url)
    if record is None:
        return None
    document_id, title, collection = record
    return CorpusDocumentMatch(document_id=document_id, title=title, collection=collection)


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
        "suggest_subject_values",
    }


def test_native_web_search_is_opted_in_only_for_responses_models() -> None:
    assert model_supports_native_web_search("openai-responses:gpt-5.2")
    assert not model_supports_native_web_search("gpt-5.2")


def test_responses_prefix_selects_transport_without_enabling_web_search(
    tool_service: CorpusToolService, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_models: list[str] = []

    def responses_model(model_name: str, *, provider: object) -> TestModel:
        selected_models.append(model_name)
        return TestModel()

    def chat_model(model_name: str, *, provider: object) -> TestModel:
        pytest.fail(f"Responses model was passed to chat transport: {model_name}")

    monkeypatch.setattr(runtime, "OpenAIResponsesModel", responses_model)
    monkeypatch.setattr(runtime, "OpenAIChatModel", chat_model)

    create_research_agent(
        ProviderConfig(
            base_url="https://provider.example/v1",
            model="openai-responses:Qwen3.8-27B-oQ4e-mtp",
            api_key="test-key",
        ),
        tool_service,
    )

    assert selected_models == ["Qwen3.8-27B-oQ4e-mtp"]


def test_provider_neutral_web_search_is_opt_in(tool_service: CorpusToolService) -> None:
    model = TestModel(custom_output_text="background", call_tools=[])
    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        tool_service,
        model=model,
        enable_web_search=True,
    )
    agent.run_sync("Describe the corpus.", deps=CorpusToolDeps(service=tool_service))
    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert any(tool.name == "search_web_terminology" for tool in parameters.function_tools)


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


def test_output_validator_teaches_a_real_but_unreturned_citation() -> None:
    known = {"https://papyri.info/ddbdp/p.mich;8;480"}
    unseen = "https://papyri.info/ddbdp/c.pap.gr;1;19"
    lookup = lambda url: _match(  # noqa: E731 - small stub
        url, {unseen: ("ddbdp:DDbDP/20/20699.xml", "c.pap.gr.1.19", "ddbdp")}
    )

    with pytest.raises(ModelRetry) as error:
        validate_research_output(f"Corpus evidence: {unseen}.", known, citation_lookup=lookup)

    assert "c.pap.gr.1.19" in error.value.message
    assert "no corpus tool returned it" in error.value.message
    assert "Search or inspect that document first" in error.value.message


def test_output_validator_teaches_a_constructed_citation_with_series_suggestions() -> None:
    known = {
        "https://papyri.info/ddbdp/c.pap.gr;1;7",
        "https://papyri.info/ddbdp/c.pap.gr;1;10",
    }
    constructed = "https://papyri.info/ddbdp/c.pap.gr;1;999"

    with pytest.raises(ModelRetry) as error:
        validate_research_output(
            f"Corpus evidence: {constructed}.", known, citation_lookup=lambda url: None
        )

    message = error.value.message
    assert "not a corpus document URL" in message
    assert "never construct" in message
    assert "Known citations for this series" in message
    assert "c.pap.gr;1;7" in message


def test_output_validator_lists_every_unknown_citation() -> None:
    with pytest.raises(ModelRetry) as error:
        validate_research_output(
            "Corpus evidence: https://papyri.info/ddbdp/a;1;1 and https://papyri.info/ddbdp/b;2;2.",
            set(),
            citation_lookup=lambda url: None,
        )

    message = error.value.message
    assert "a;1;1" in message
    assert "b;2;2" in message


def test_output_validator_tolerates_attached_punctuation_and_invisible_characters() -> None:
    known = {"https://papyri.info/ddbdp/p.mich;8;480"}
    answer = (
        "Corpus evidence: https://papyri.info/ddbdp/p.mich;8;480; then "
        "“https://papyri.info/ddbdp/p.mich;8;480” and "
        "https://papyri.info/ddbdp/p.mich;8;480\u200b done."
    )

    assert validate_research_output(answer, known) == answer
