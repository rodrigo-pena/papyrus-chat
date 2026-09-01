"""Structured, distinct-document retrieval over the v2 artifact."""

import sqlite3
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


def _insert_search_document(
    search: StructuredCorpusSearch,
    document_id: str,
    *,
    title: str = "Synthetic",
    metadata: str = "{}",
    passage_text: str | None = None,
) -> None:
    connection = search._connection  # noqa: SLF001 - focused ranking fixture
    connection.execute(
        "INSERT INTO documents VALUES (?, 'dclp', ?, '[\"grc\"]', ?, 'repo', 'commit', "
        "?, NULL, NULL)",
        (document_id, title, metadata, document_id),
    )
    connection.execute(
        "INSERT INTO documents_fts (document_id, title, metadata) VALUES (?, ?, ?)",
        (document_id, title, metadata),
    )
    if passage_text is None:
        return
    passage_id = f"{document_id}#edition:1:edition"
    connection.execute(
        "INSERT INTO passages VALUES (?, ?, 'edition', 1, NULL, NULL, ?, ?, '{}', "
        "'repo', 'commit', ?, 'edition')",
        (passage_id, document_id, passage_text, passage_text.casefold(), document_id),
    )
    connection.execute("INSERT INTO passage_languages VALUES (?, 'grc')", (passage_id,))
    connection.execute(
        "INSERT INTO passages_fts (search_text, title, passage_id) VALUES (?, ?, ?)",
        (passage_text.casefold(), title, passage_id),
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


@pytest.fixture()
def literary_search(tmp_path: Path, fixture_git_repo: Path) -> StructuredCorpusSearch:
    artifact = tmp_path / "corpus"
    build_artifact(
        ["dclp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )
    return StructuredCorpusSearch(artifact / "corpus.sqlite")


@pytest.fixture()
def mixed_search(tmp_path: Path, fixture_git_repo: Path) -> StructuredCorpusSearch:
    artifact = tmp_path / "corpus"
    build_artifact(
        ["dclp", "ddbdp"],
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


def test_query_accepts_the_whole_query_as_json_text() -> None:
    query = CorpusQuery.model_validate(
        '{"collections": ["ddbdp"], "term_groups": [["Κλαύδιος"]], '
        '"transcription_languages": ["grc"], '
        '"date_interval": {"not_before": 100, "not_after": 125}, "limit": 5}'
    )

    assert query.collections == ("ddbdp",)
    assert query.term_groups == (("Κλαύδιος",),)
    assert query.transcription_languages == ("grc",)
    assert query.date_interval == CorpusDateInterval(not_before=100, not_after=125)
    assert query.limit == 5


def test_query_decodes_stringified_nested_arguments() -> None:
    query = CorpusQuery.model_validate(
        {
            "term_groups": '[["Κλαύδιος", "Claudius"], ["Geld"]]',
            "fields": '["transcription", "metadata"]',
            "transcription_languages": '"grc"',
            "date_interval": '{"not_before": 100, "not_after": 125}',
        }
    )

    assert query.term_groups == (("Κλαύδιος", "Claudius"), ("Geld",))
    assert query.fields == ("transcription", "metadata")
    assert query.transcription_languages == ("grc",)
    assert query.date_interval == CorpusDateInterval(not_before=100, not_after=125)


def test_query_recovers_members_swallowed_into_a_term_groups_string() -> None:
    swallowed = (
        '[["Κλαύδιος", "Claudius"], ["Geld", "δραχμή"]],\n'
        ' "transcription_languages": ["grc"], '
        '"date_interval": {"not_before": 100, "not_after": 125}}'
    )

    query = CorpusQuery.model_validate({"term_groups": swallowed, "fields": ["transcription"]})

    assert query.term_groups == (("Κλαύδιος", "Claudius"), ("Geld", "δραχμή"))
    assert query.fields == ("transcription",)
    assert query.transcription_languages == ("grc",)
    assert query.date_interval == CorpusDateInterval(not_before=100, not_after=125)


def test_query_rejects_unknown_argument_keys() -> None:
    with pytest.raises(ValidationError):
        CorpusQuery.model_validate({"term_groups": [["Κλαύδιος"]], "query": "Claudius"})


def test_term_matching_does_not_rescan_full_text_indexes_per_document(
    documentary_search: StructuredCorpusSearch,
) -> None:
    query = CorpusQuery(
        term_groups=[["Κλαύδιος", "Claudius"], ["Geld"]], fields=["transcription", "metadata"]
    )
    where, params = documentary_search._where_clause(query)  # noqa: SLF001 - plan fixture
    plan = documentary_search._connection.execute(  # noqa: SLF001 - plan fixture
        f"EXPLAIN QUERY PLAN SELECT count(*) FROM documents d WHERE {' AND '.join(where)}",
        params,
    ).fetchall()
    parents = {row[0]: row[1] for row in plan}
    details = {row[0]: row[3] for row in plan}

    def has_correlated_ancestor(node: int) -> bool:
        while (parent := parents.get(node)) is not None:
            if details.get(parent, "").startswith("CORRELATED"):
                return True
            node = parent
        return False

    fts_scans = [row[0] for row in plan if "VIRTUAL TABLE" in row[3]]
    assert fts_scans
    assert not any(has_correlated_ancestor(node) for node in fts_scans)


def test_language_filter_probes_documents_instead_of_scanning_languages(
    documentary_search: StructuredCorpusSearch,
) -> None:
    query = CorpusQuery(term_groups=[["Κλαύδιος"]], transcription_languages=["grc"])
    where, params = documentary_search._where_clause(query)  # noqa: SLF001 - plan fixture
    plan = documentary_search._connection.execute(  # noqa: SLF001 - plan fixture
        f"EXPLAIN QUERY PLAN SELECT count(*) FROM documents d WHERE {' AND '.join(where)}",
        params,
    ).fetchall()
    details = [row[3] for row in plan]

    # 'grc' matches nearly every passage language row, so driving the correlated
    # filter from the language index re-probes that table per candidate document.
    assert any(
        "SEARCH p USING INDEX passages_by_document (document_id=?)" in detail for detail in details
    )
    assert not any("passage_languages_lookup (language" in detail for detail in details)


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
    assert hit.passage_language == "grc"
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
    hgv = next(
        component for component in metadata_result.hits[0].components if component.kind == "hgv"
    )
    assert {"Geld", "erneute Petition"} <= set(hgv.metadata["subject"])
    assert hgv.dates[0].not_before == "0101"
    assert hgv.dates[0].not_after == "0125"
    assert hgv.source.path == "HGV_meta_EpiDoc/HGV28/27093.xml"

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


def test_facets_aggregate_without_binding_every_candidate_document(
    documentary_search: StructuredCorpusSearch,
) -> None:
    connection = documentary_search._connection  # noqa: SLF001 - SQLite limit regression
    source = connection.execute("SELECT * FROM documents LIMIT 1").fetchone()
    for index in range(12):
        values = list(source)
        values[0] = f"ddbdp:synthetic-{index}"
        connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
    previous_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
    try:
        facets = documentary_search.facet_documents(CorpusQuery(), "collection")
    finally:
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    assert facets.values == (CorpusFacetValue(value="ddbdp", count=13),)


def test_facets_count_mixed_collections_and_actual_passage_languages(
    mixed_search: StructuredCorpusSearch,
) -> None:
    collections = mixed_search.facet_documents(CorpusQuery(), "collection")
    languages = mixed_search.facet_documents(CorpusQuery(collections=["dclp"]), "language")

    assert collections.values == (
        CorpusFacetValue(value="dclp", count=2),
        CorpusFacetValue(value="ddbdp", count=1),
    )
    assert languages.values == (CorpusFacetValue(value="grc", count=1),)


def test_describe_reports_distinct_corpus_inventory(
    documentary_search: StructuredCorpusSearch,
) -> None:
    description = documentary_search.describe()

    assert description.collections == ("ddbdp",)
    assert description.documents == 1
    assert description.passages == 1
    assert description.components == 2
    assert description.languages == ("grc",)


def test_transcription_languages_use_actual_edition_passages(
    literary_search: StructuredCorpusSearch,
) -> None:
    english = literary_search.query(
        CorpusQuery(
            term_groups=[["ἔτους"]],
            fields=["transcription"],
            transcription_languages=["en"],
        )
    )
    greek = literary_search.query(
        CorpusQuery(
            term_groups=[["ἔτους"]],
            fields=["transcription"],
            transcription_languages=["grc"],
        )
    )
    either = literary_search.query(
        CorpusQuery(
            term_groups=[["ἔτους"]],
            fields=["transcription"],
            transcription_languages=["en", "grc"],
        )
    )

    assert english.candidate_count == 0
    assert greek.candidate_count == 1
    assert either.candidate_count == 1
    assert literary_search.describe().languages == ("grc",)


@pytest.mark.parametrize(
    ("not_before", "not_after", "when_value", "query_bounds", "expected"),
    [
        ("0101", "0125", None, (110, 110), 1),
        (None, None, "0110", (110, 110), 1),
        ("0101", None, None, (300, 350), 1),
        (None, "0125", None, (-300, -200), 1),
        ("-0200", "-0100", None, (-150, -150), 1),
        (None, None, None, (100, 125), 0),
    ],
)
def test_date_filters_preserve_open_ended_and_point_intervals(
    documentary_search: StructuredCorpusSearch,
    not_before: str | None,
    not_after: str | None,
    when_value: str | None,
    query_bounds: tuple[int, int],
    expected: int,
) -> None:
    documentary_search._connection.execute(  # noqa: SLF001 - focused database regression
        "UPDATE dates SET not_before = ?, not_after = ?, when_value = ?",
        (not_before, not_after, when_value),
    )

    result = documentary_search.query(
        CorpusQuery(
            date_interval=CorpusDateInterval(not_before=query_bounds[0], not_after=query_bounds[1])
        )
    )

    assert result.candidate_count == expected


def test_inspect_documents_preserves_requested_order_and_bounds_passages(
    documentary_search: StructuredCorpusSearch,
) -> None:
    inspections = documentary_search.inspect_documents(
        ["missing", "ddbdp:DDbDP/27/27093.xml"], excerpt_limit=1
    )

    assert [inspection.document_id for inspection in inspections] == ["ddbdp:DDbDP/27/27093.xml"]
    assert len(inspections[0].passages) == 1
    assert inspections[0].passages[0].line_reference == "lines 1-18"
    assert {component.kind for component in inspections[0].components} == {"ddbdp", "hgv"}


def test_query_is_safe_for_fts_injection_and_reports_truncation(
    documentary_search: StructuredCorpusSearch,
) -> None:
    result = documentary_search.query(
        CorpusQuery(term_groups=[['" OR * NOT'], ["Κλαύδιος"]], limit=1)
    )

    assert isinstance(result, CorpusSearchResult)
    assert result.candidate_count == 0
    assert result.truncated is False


def test_query_ranks_relevance_before_alphabetical_document_ids(
    documentary_search: StructuredCorpusSearch,
) -> None:
    _insert_search_document(
        documentary_search, "dclp:a-metadata", metadata='{"subject":"rankingneedle"}'
    )
    _insert_search_document(documentary_search, "dclp:m-title", title="rankingneedle")
    _insert_search_document(
        documentary_search,
        "dclp:z-passage",
        passage_text="rankingneedle rankingneedle rankingneedle",
    )

    result = documentary_search.query(CorpusQuery(term_groups=[["rankingneedle"]], limit=2))

    assert result.candidate_count == 3
    assert result.truncated is True
    assert [hit.document_id for hit in result.hits] == ["dclp:m-title", "dclp:a-metadata"]


def test_query_ranks_stronger_passage_match_before_alphabetical_id(
    documentary_search: StructuredCorpusSearch,
) -> None:
    _insert_search_document(documentary_search, "dclp:a-passage", passage_text="textneedle")
    _insert_search_document(
        documentary_search,
        "dclp:z-passage",
        passage_text="textneedle textneedle textneedle textneedle",
    )

    result = documentary_search.query(
        CorpusQuery(term_groups=[["textneedle"]], fields=["transcription"], limit=1)
    )

    assert result.candidate_count == 2
    assert result.truncated is True
    assert result.hits[0].document_id == "dclp:z-passage"


def test_query_uses_deterministic_ties_and_alphabetical_nonlexical_order(
    documentary_search: StructuredCorpusSearch,
) -> None:
    _insert_search_document(documentary_search, "dclp:z-tie", passage_text="tiephrase")
    _insert_search_document(documentary_search, "dclp:a-tie", passage_text="tiephrase")

    lexical = documentary_search.query(
        CorpusQuery(term_groups=[["tiephrase"]], fields=["transcription"])
    )
    nonlexical = documentary_search.query(CorpusQuery())

    assert [hit.document_id for hit in lexical.hits] == ["dclp:a-tie", "dclp:z-tie"]
    assert [hit.document_id for hit in nonlexical.hits] == sorted(
        hit.document_id for hit in nonlexical.hits
    )
