"""Transport-neutral corpus service lifecycle tests."""

import sqlite3
from pathlib import Path

import pytest

from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.structured import CorpusQuery


def test_open_binds_a_read_only_sqlite_connection(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    try:
        assert service.artifact_root == corpus_artifact.resolve()
        assert service.manifest.collections == ["dclp", "translations"]
        with pytest.raises(sqlite3.OperationalError):
            service._connection.execute("CREATE TABLE should_not_exist (value TEXT)")
    finally:
        service.close()


def test_close_releases_the_service_connection(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    service.close()

    with pytest.raises(sqlite3.ProgrammingError):
        service._connection.execute("SELECT 1")


def test_get_corpus_info_reports_manifest_provenance_and_capability(
    corpus_artifact: Path,
) -> None:
    service = CorpusService.open(corpus_artifact)

    try:
        info = service.get_corpus_info()
        assert info.artifact_schema_version == 3
        assert info.builder.name == "papyrus-corpus-build"
        assert info.source.resolved_commit
        assert info.collections == ("dclp", "translations")
        assert info.statistics.documents == 4
        assert info.languages == ("grc",)
        assert info.logical_content_hash.startswith("sha256:")
        assert info.semantic_capability.available is False
        assert "semantic" in (info.semantic_capability.unavailable_reason or "")
    finally:
        service.close()


def test_lookup_document_normalizes_input_and_reports_empty_matches(
    corpus_artifact: Path,
) -> None:
    service = CorpusService.open(corpus_artifact)

    try:
        found = service.lookup_document(" tm:23944 ")
        assert found.normalized_identifier == "tm:23944"
        assert found.exact_match_count == 1
        assert found.truncated is False
        assert found.matches[0].document_id == "dclp:DCLP/23/23944.xml"

        missing = service.lookup_document("TM 999999")
        assert missing.exact_match_count == 0
        assert missing.matches == ()
        assert missing.truncated is False
    finally:
        service.close()


def test_facets_are_sql_bounded_and_report_exact_total_values(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    try:
        result = service.facet_documents(CorpusQuery(), "collection", limit=1)
        assert len(result.values) == 1
        assert result.total_values == 2
        assert result.truncated is True
        assert result.limit == 1
    finally:
        service.close()


def test_inspection_requires_at_least_one_requested_id(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    try:
        with pytest.raises(ValueError, match="at least 1"):
            service.inspect_documents([])
    finally:
        service.close()
