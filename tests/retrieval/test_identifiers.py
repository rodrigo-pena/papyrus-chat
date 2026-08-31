"""Normalized exact identifier lookup (SPEC 8)."""

import sqlite3
from pathlib import Path

from papyrus_chat.retrieval.identifiers import IdentifierLookup, normalize_identifier_query
from papyrus_chat.textnorm import normalize_identifier_value


def lookup(corpus_artifact: Path) -> IdentifierLookup:
    return IdentifierLookup(corpus_artifact / "corpus.sqlite")


class TestNormalization:
    def test_case_and_whitespace_are_normalized(self) -> None:
        assert normalize_identifier_query("  TM 23944 ") == ("tm", "23944")
        assert normalize_identifier_query("tm:23944") == ("tm", "23944")
        assert normalize_identifier_query("TM-23944") == ("tm", "23944")

    def test_bare_value_has_no_namespace(self) -> None:
        assert normalize_identifier_query("23944") == ("", "23944")

    def test_dotted_values_are_not_split(self) -> None:
        assert normalize_identifier_query("p.tebt.1.7") == ("", "p.tebt.1.7")

    def test_zero_width_characters_are_stripped(self) -> None:
        assert normalize_identifier_query("23\u200b944") == ("", "23944")
        assert normalize_identifier_value("23\u200b944") == "23944"


class TestLookup:
    def test_canonical_and_messy_forms_find_the_document(self, corpus_artifact: Path) -> None:
        service = lookup(corpus_artifact)

        for query in ("TM 23944", "tm:23944", "23944"):
            results = service.lookup(query)
            assert results, f"no results for {query!r}"
            assert any(doc.document_id == "dclp:DCLP/23/23944.xml" for doc in results)

    def test_messy_ddbdp_value_is_found(self, corpus_artifact: Path) -> None:
        service = lookup(corpus_artifact)

        results = service.lookup("P.TEBT.1.7")

        assert any(doc.document_id.endswith("3643-1.xml") for doc in results)

    def test_metadata_only_documents_are_returned(self, corpus_artifact: Path) -> None:
        service = lookup(corpus_artifact)

        results = service.lookup("23702")

        assert any(doc.document_id == "dclp:DCLP/23/23702.xml" for doc in results)
        assert all(len(doc.languages) >= 0 for doc in results)

    def test_unknown_identifier_returns_empty(self, corpus_artifact: Path) -> None:
        service = lookup(corpus_artifact)

        assert service.lookup("999999") == []
        assert service.lookup("TM 999999") == []

    def test_results_are_deterministically_ordered(self, corpus_artifact: Path) -> None:
        service = lookup(corpus_artifact)

        first = service.lookup("23702")
        second = service.lookup("23702")

        assert [d.document_id for d in first] == [d.document_id for d in second]

    def test_lookup_uses_the_index(self, corpus_artifact: Path) -> None:
        connection = sqlite3.connect(corpus_artifact / "corpus.sqlite")
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT document_id FROM identifiers"
            " WHERE namespace_norm = ? AND value_norm = ?",
            ("tm", "23944"),
        ).fetchall()
        connection.close()

        plan_text = " ".join(str(row) for row in plan)
        assert "USING INDEX" in plan_text or "SEARCH" in plan_text
        assert "SCAN identifiers" not in plan_text
