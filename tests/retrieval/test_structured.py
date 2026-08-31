"""Structured, distinct-document retrieval over the v2 artifact."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.retrieval.structured import (
    CorpusDateInterval,
    CorpusFacetValue,
    CorpusQuery,
    CorpusSearchResult,
    StructuredCorpusSearch,
)


@pytest.fixture()
def documentary_search(tmp_path: Path, fixture_git_repo: Path) -> StructuredCorpusSearch:
    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )
    return StructuredCorpusSearch(artifact / "corpus.sqlite")


def test_query_normalizes_filters_and_preserves_lexical_groups() -> None:
    query = CorpusQuery(
        collections=["DDBDP", "dclp"],
        term_groups=[[" taxes ", "Fiscal"], ["letter"]],
        fields=["translation", "transcription"],
        transcription_languages=["GRC"],
        date_interval=CorpusDateInterval(not_before=100, not_after=125),
        limit=7,
    )

    assert query.collections == ("dclp", "ddbdp")
    assert query.term_groups == (("taxes", "Fiscal"), ("letter",))
    assert query.fields == ("translation", "transcription")
    assert query.transcription_languages == ("grc",)
    assert query.date_interval == CorpusDateInterval(not_before=100, not_after=125)
    assert query.limit == 7


def test_query_rejects_unbounded_or_empty_groups() -> None:
    with pytest.raises(ValidationError):
        CorpusQuery(term_groups=[[]])
    with pytest.raises(ValidationError):
        CorpusQuery(limit=101)


def test_transcription_query_returns_exact_distinct_candidates_and_evidence(
    documentary_search: StructuredCorpusSearch,
) -> None:
    result = documentary_search.query(
        CorpusQuery(
            collections=["ddbdp"],
            term_groups=[["Κλαύδιος", "not-present"], ["πατήρ"]],
            fields=["transcription"],
            transcription_languages=["grc"],
            date_interval=CorpusDateInterval(not_before=100, not_after=125),
            limit=1,
        )
    )

    assert isinstance(result, CorpusSearchResult)
    assert result.query.collections == ("ddbdp",)
    assert result.query.term_groups == (("Κλαύδιος", "not-present"), ("πατήρ",))
    assert result.candidate_count == 1
    assert result.truncated is False
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.document_id == "ddbdp:DDbDP/27/27093.xml"
    assert hit.passage_kind == "edition"
    assert hit.passage_text
    assert hit.line_reference == "lines 1-18"
    assert hit.source.path == "DDbDP/27/27093.xml"
    assert hit.canonical_url == "https://papyri.info/ddbdp/p.mich;8;480"


def test_metadata_and_date_filters_use_linked_hgv_data(
    documentary_search: StructuredCorpusSearch,
) -> None:
    metadata_result = documentary_search.query(
        CorpusQuery(term_groups=[["Geld"]], fields=["metadata"])
    )
    assert metadata_result.candidate_count == 1
    assert metadata_result.hits[0].document_id == "ddbdp:DDbDP/27/27093.xml"

    title_only = documentary_search.query(CorpusQuery(term_groups=[["Geld"]], fields=["title"]))
    assert title_only.candidate_count == 0

    metadata_only = documentary_search.query(
        CorpusQuery(term_groups=[["Terentianus to"]], fields=["metadata"])
    )
    assert metadata_only.candidate_count == 0

    outside_date = documentary_search.query(
        CorpusQuery(
            term_groups=[["Geld"]],
            fields=["metadata"],
            date_interval=CorpusDateInterval(not_before=200, not_after=300),
        )
    )
    assert outside_date.candidate_count == 0
    assert outside_date.hits == ()

    wrong_language = documentary_search.query(
        CorpusQuery(
            term_groups=[["Κλαύδιος"]],
            fields=["transcription"],
            transcription_languages=["de"],
        )
    )
    assert wrong_language.candidate_count == 0

    title_result = documentary_search.query(
        CorpusQuery(term_groups=[["Terentianus to"]], fields=["title"])
    )
    assert title_result.candidate_count == 1


def test_facets_count_distinct_documents_from_the_normalized_query(
    documentary_search: StructuredCorpusSearch,
) -> None:
    facets = documentary_search.facet_documents(CorpusQuery(), "subject")

    assert facets.query == CorpusQuery()
    assert facets.field == "subject"
    assert CorpusFacetValue(value="Geld", count=1) in facets.values


def test_query_is_safe_for_fts_injection_and_reports_truncation(
    documentary_search: StructuredCorpusSearch,
) -> None:
    result = documentary_search.query(
        CorpusQuery(term_groups=[['" OR * NOT'], ["Κλαύδιος"]], limit=1)
    )

    assert isinstance(result, CorpusSearchResult)
    assert result.candidate_count == 0
    assert result.truncated is False
