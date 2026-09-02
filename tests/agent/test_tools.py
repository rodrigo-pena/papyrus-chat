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
    _hit_summary,  # noqa: PLC2701 - projection unit test
    _inspection_outcome,  # noqa: PLC2701 - projection unit test
    _inspection_summaries,  # noqa: PLC2701 - projection unit test
    _search_summary,  # noqa: PLC2701 - projection unit test
    register_corpus_tools,
)
from papyrus_chat.artifact.records import (
    ComponentDateRecord,
    ComponentIdentifierRecord,
    ComponentRecord,
    SourceReference,
)
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.retrieval.structured import (
    CorpusHit,
    CorpusInspection,
    CorpusQuery,
    CorpusSearchResult,
    StructuredCorpusSearch,
)


def _source() -> SourceReference:
    return SourceReference(
        repository_url="https://github.com/papyri/idp.data.git",
        commit="ffc23d0",
        path="DDbDP/41/41819.xml",
    )


def _hit(*, text: str) -> CorpusHit:
    return CorpusHit(
        document_id="ddbdp:DDbDP/41/41819.xml",
        title="psi.congr.xvii.22",
        collection="ddbdp",
        languages=("grc",),
        metadata={"authority": "Duke Collaboratory for Classics Computing (DC3)"},
        passage_id="ddbdp:DDbDP/41/41819.xml#edition:2:r,1",
        passage_kind="edition",
        passage_language="grc",
        passage_text=text,
        snippet=text[:200],
        line_reference="lines 1-13",
        components=(
            ComponentRecord(
                component_id="ddbdp:DDbDP/41/41819.xml",
                document_id="ddbdp:DDbDP/41/41819.xml",
                kind="ddbdp",
                title="psi.congr.xvii.22",
                identifiers=(
                    ComponentIdentifierRecord(
                        component_id="ddbdp:DDbDP/41/41819.xml",
                        namespace="HGV",
                        value="41819",
                    ),
                ),
                source=_source(),
            ),
            ComponentRecord(
                component_id="hgv:41819",
                document_id=None,
                kind="hgv",
                title="Conto privato",
                metadata={"subject": ("Abrechnung", "privat"), "material": ("Papyrus",)},
                dates=(
                    ComponentDateRecord(
                        component_id="hgv:41819",
                        sequence=0,
                        text="nach 19. Jan. 114 v.Chr.",
                    ),
                ),
                source=_source(),
            ),
        ),
        source=_source(),
        canonical_url="https://papyri.info/ddbdp/psi.congr.xvii;;22",
    )


def test_search_summary_keeps_identity_evidence_and_drops_heavy_fields() -> None:
    result = CorpusSearchResult(
        query=CorpusQuery(term_groups=[["χρέος"]]),
        candidate_count=1,
        truncated=False,
        hits=(_hit(text="λόγος " * 30),),
    )

    dumped = _search_summary(result).model_dump()

    hit = dumped["hits"][0]
    assert hit["snippet"].startswith("λόγος")
    assert hit["canonical_url"] == "https://papyri.info/ddbdp/psi.congr.xvii;;22"
    assert hit["line_reference"] == "lines 1-13"
    assert "passage_text" not in hit
    assert "components" not in hit
    assert "source" not in hit
    assert "identifiers" not in str(dumped)
    assert "passage_id" not in hit


def test_search_summary_carries_group_candidate_counts() -> None:
    result = CorpusSearchResult(
        query=CorpusQuery(term_groups=[["Geld"], ["zzz-absent"]]),
        candidate_count=0,
        truncated=False,
        hits=(),
        group_candidate_counts=(1, 0),
    )

    summary = _search_summary(result)

    assert summary.group_candidate_counts == (1, 0)


def test_inspection_summary_truncates_excerpt_and_keeps_hgv_context() -> None:
    inspection = CorpusInspection(
        document_id="ddbdp:DDbDP/41/41819.xml",
        title="psi.congr.xvii.22",
        collection="ddbdp",
        languages=("grc",),
        metadata={"authority": "Duke Collaboratory for Classics Computing (DC3)"},
        source=_source(),
        canonical_url="https://papyri.info/ddbdp/psi.congr.xvii;;22",
        components=_hit(text="").components,
        passages=(_hit(text="ὀφείλω " * 200),),
    )

    (summary,) = _inspection_summaries((inspection,))

    assert summary.hgv is not None
    assert summary.hgv.metadata["subject"] == ("Abrechnung", "privat")
    assert summary.hgv.date_texts == ("nach 19. Jan. 114 v.Chr.",)
    assert summary.passages[0].excerpt is not None
    assert summary.passages[0].excerpt.endswith("…")
    assert len(summary.passages[0].excerpt) <= 501
    dumped = summary.model_dump()
    assert "identifiers" not in str(dumped)
    assert "repository_url" not in str(dumped)


def test_inspection_excerpt_centers_on_a_focus_term() -> None:
    text = "λόγος ἀναγραφὴς οἴνου " * 200 + "χρέος ἀπόδος τοῦ ἀρταβάνου " + "καὶ ἄλλο " * 100
    inspection = CorpusInspection(
        document_id="ddbdp:DDbDP/41/41819.xml",
        title="psi.congr.xvii.22",
        collection="ddbdp",
        languages=("grc",),
        metadata={},
        source=_source(),
        canonical_url="https://papyri.info/ddbdp/psi.congr.xvii;;22",
        components=(),
        passages=(_hit(text=text),),
    )

    default = next(
        summary.passages[0].excerpt
        for summary in _inspection_summaries((inspection,))
        if summary.passages
    )
    focused = next(
        summary.passages[0].excerpt
        for summary in _inspection_summaries((inspection,), focus_terms=("χρεο",))
        if summary.passages
    )

    assert default is not None and "χρέος" not in default
    assert focused is not None and "χρέος" in focused
    assert focused.startswith("…")


def test_inspection_excerpt_honors_a_larger_excerpt_budget() -> None:
    text = "λόγος ἀναγραφὴς οἴνου " * 100
    inspection = CorpusInspection(
        document_id="ddbdp:DDbDP/41/41819.xml",
        title="psi.congr.xvii.22",
        collection="ddbdp",
        languages=("grc",),
        metadata={},
        source=_source(),
        canonical_url="https://papyri.info/ddbdp/psi.congr.xvii;;22",
        components=(),
        passages=(_hit(text=text),),
    )

    (summary,) = _inspection_summaries((inspection,), excerpt_chars=2000)

    excerpt = summary.passages[0].excerpt
    assert excerpt is not None
    assert len(excerpt) > 501
    assert len(excerpt) <= 2002


def test_short_excerpt_is_not_truncated() -> None:
    assert _hit_summary(_hit(text="βραχύ"))  # sanity: summary built
    excerpt_text = "ἀργύριον τὸ δοσόν"
    inspection = CorpusInspection(
        document_id="ddbdp:DDbDP/2/2914.xml",
        title="p.petr.3.43_2",
        collection="ddbdp",
        languages=("grc",),
        metadata={},
        source=_source(),
        canonical_url="https://papyri.info/ddbdp/p.petr;3;43_2",
        components=(),
        passages=(_hit(text=excerpt_text),),
    )

    (summary,) = _inspection_summaries((inspection,))

    assert summary.passages[0].excerpt == excerpt_text
    assert summary.hgv is None


def test_inspection_outcome_reports_missing_document_ids() -> None:
    inspection = CorpusInspection(
        document_id="ddbdp:DDbDP/27/27093.xml",
        title="p.mich.8.480",
        collection="ddbdp",
        languages=("grc",),
        metadata={},
        source=_source(),
        canonical_url="https://papyri.info/ddbdp/p.mich;8;480",
        components=(),
        passages=(),
    )
    requested = [
        "ddbdp:DDbDP/27/27093.xml",
        "ddbdp:DDbDP/99/99999.xml",
        "ddbdp:DDbDP/99/99999.xml",
    ]

    outcome = _inspection_outcome((inspection,), requested)

    assert [summary.document_id for summary in outcome.inspections] == ["ddbdp:DDbDP/27/27093.xml"]
    assert outcome.missing == ("ddbdp:DDbDP/99/99999.xml",)


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
    facets = shared_hgv_tools.facet_documents(CorpusQuery(), "subject")
    assert any(value.value == "Geld" and value.count == 2 for value in facets.values)


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
        "suggest_subject_values",
    }


def test_tool_schemas_state_inspection_bounds_and_facet_options(
    corpus_tools: CorpusToolService,
) -> None:
    model = TestModel()
    agent = Agent(model, deps_type=CorpusToolDeps)
    register_corpus_tools(agent)

    agent.run_sync("Describe the corpus.", deps=CorpusToolDeps(service=corpus_tools))

    parameters = model.last_model_request_parameters
    assert parameters is not None
    schemas = {tool.name: tool.parameters_json_schema for tool in parameters.function_tools}

    inspection = schemas["inspect_documents"]["properties"]
    assert inspection["document_ids"]["maxItems"] == 20
    assert inspection["excerpt_limit"]["minimum"] == 1
    assert inspection["excerpt_limit"]["maximum"] == 10
    assert inspection["excerpt_limit"]["default"] == 3
    assert inspection["excerpt_chars"]["minimum"] == 200
    assert inspection["excerpt_chars"]["maximum"] == 2000
    assert inspection["excerpt_chars"]["default"] == 500
    focus = inspection["focus_terms"]
    assert focus["maxItems"] == 8
    assert focus["items"]["minLength"] == 1
    assert focus["items"]["maxLength"] == 200

    field = schemas["facet_documents"]["properties"]["field"]
    assert set(field["enum"]) == {"collection", "language", "subject", "material", "origin", "kind"}
    assert "HGV component metadata" in field["description"]
