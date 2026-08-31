"""Typed, read-only Pydantic AI corpus tools."""

import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from papyrus_chat.agent.tools import (
    CorpusToolDeps,
    CorpusToolService,
    register_corpus_tools,
)
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.retrieval.structured import CorpusQuery, StructuredCorpusSearch


@pytest.fixture()
def corpus_tools(tmp_path: Path, fixture_git_repo: Path) -> CorpusToolService:
    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )
    return CorpusToolService(StructuredCorpusSearch(artifact / "corpus.sqlite"))


@pytest.fixture()
def shared_hgv_tools(tmp_path: Path, fixture_git_repo: Path) -> CorpusToolService:
    source_repo = tmp_path / "source"
    subprocess.run(["git", "clone", "--quiet", str(fixture_git_repo), str(source_repo)], check=True)
    duplicate = source_repo / "DDbDP" / "27" / "27094.xml"
    shutil.copy2(source_repo / "DDbDP" / "27" / "27093.xml", duplicate)
    subprocess.run(["git", "add", str(duplicate)], cwd=source_repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "add shared HGV fixture",
        ],
        cwd=source_repo,
        check=True,
    )
    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(source_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="HEAD",
    )
    return CorpusToolService(StructuredCorpusSearch(artifact / "corpus.sqlite"))


def test_describe_corpus_reports_inventory(corpus_tools: CorpusToolService) -> None:
    description = corpus_tools.describe_corpus()

    assert description.collections == ("ddbdp",)
    assert description.documents == 1
    assert description.passages == 1
    assert description.components == 2
    assert description.languages == ("grc",)


def test_search_tool_returns_the_complete_query_and_assumptions(
    corpus_tools: CorpusToolService,
) -> None:
    result = corpus_tools.search_documents(
        CorpusQuery(term_groups=[["Κλαύδιος"]], fields=["transcription"]),
        assumptions=("The requested period was interpreted as the linked HGV date range.",),
    )

    assert result.query.term_groups == (("Κλαύδιος",),)
    assert result.assumptions == (
        "The requested period was interpreted as the linked HGV date range.",
    )
    assert result.candidate_count == 1
    assert result.hits[0].canonical_url == "https://papyri.info/ddbdp/p.mich;8;480"
    hgv = next(component for component in result.hits[0].components if component.kind == "hgv")
    assert {"Geld", "erneute Petition"} <= set(hgv.metadata["subject"])
    assert hgv.dates[0].text == "frühes II"
    assert hgv.source.path == "HGV_meta_EpiDoc/HGV28/27093.xml"


def test_inspect_tool_is_bounded_and_returns_located_passages(
    corpus_tools: CorpusToolService,
) -> None:
    result = corpus_tools.inspect_documents(["ddbdp:DDbDP/27/27093.xml"], excerpt_limit=1)

    assert len(result.inspections) == 1
    assert len(result.inspections[0].passages) == 1
    assert result.inspections[0].passages[0].line_reference == "lines 1-18"
    assert {component.kind for component in result.inspections[0].components} == {"ddbdp", "hgv"}

    with pytest.raises(ValueError, match="at most 20"):
        corpus_tools.inspect_documents([f"missing:{index}" for index in range(21)])


def test_inspect_tool_exposes_shared_hgv_evidence_for_each_linked_document(
    shared_hgv_tools: CorpusToolService,
) -> None:
    result = shared_hgv_tools.inspect_documents(
        ["ddbdp:DDbDP/27/27093.xml", "ddbdp:DDbDP/27/27094.xml"]
    )

    assert len(result.inspections) == 2
    for inspection in result.inspections:
        hgv = next(component for component in inspection.components if component.kind == "hgv")
        assert "Geld" in hgv.metadata["subject"]
        assert hgv.dates[0].not_before == "0101"
        assert hgv.source.path == "HGV_meta_EpiDoc/HGV28/27093.xml"


def test_facet_tool_returns_typed_counts(corpus_tools: CorpusToolService) -> None:
    result = corpus_tools.facet_documents(CorpusQuery(), "subject")

    assert result.values
    assert any(value.value == "Geld" and value.count == 1 for value in result.values)


def test_tools_register_with_pydantic_ai_and_keep_read_only_names(
    corpus_tools: CorpusToolService,
) -> None:
    model = TestModel()
    agent = Agent(model, deps_type=CorpusToolDeps)
    register_corpus_tools(agent)

    agent.run_sync("Describe the corpus.", deps=CorpusToolDeps(service=corpus_tools))

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert {tool.name for tool in parameters.function_tools} == {
        "describe_corpus",
        "search_documents",
        "inspect_documents",
        "facet_documents",
    }
