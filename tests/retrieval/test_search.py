"""Full-text search behavior: ordering, safety, filters (SPEC 8)."""

from pathlib import Path

from papyrus_chat.retrieval.evidence import EvidencePacket
from papyrus_chat.retrieval.search import CorpusSearch, SearchFilters


def searcher(corpus_artifact: Path) -> CorpusSearch:
    return CorpusSearch(corpus_artifact / "corpus.sqlite")


class TestSearchOrdering:
    def test_identifier_queries_take_precedence(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        packet = search.search("TM 23944")

        assert packet.strategy == "identifier"
        assert any(item.document_id == "dclp:DCLP/23/23944.xml" for item in packet.items)

    def test_free_text_uses_full_text(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        packet = search.search("sovereigns decree")

        assert packet.strategy == "full-text"
        assert packet.items
        assert any(item.kind == "translation" for item in packet.items)


class TestGreekMatching:
    def test_matches_across_case_and_diacritics(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        accented_lower = search.search("ἔτους").items
        upper_plain = search.search("ΕΤΟΥΣ").items

        assert accented_lower, "accented Greek query must match"
        assert any(a.passage_id == b.passage_id for a in accented_lower for b in upper_plain), (
            "case-folded query must match the same passages"
        )

    def test_fts_syntax_is_treated_as_literal_text(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        for hostile in ('"drop OR NEAR(', 'ab" AND (col*', "NOT NOT *"):
            packet = search.search(hostile)

            assert isinstance(packet, EvidencePacket)
            assert packet.strategy in ("identifier", "full-text")


class TestFilters:
    def test_collection_filter(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        packet = search.search("decree", SearchFilters(collection="translations"))

        assert packet.items
        assert {item.collection for item in packet.items} == {"translations"}

    def test_kind_filter(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        packet = search.search("decree", SearchFilters(kind="translation"))

        assert packet.items
        assert {item.kind for item in packet.items} == {"translation"}

    def test_document_scope_filter(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        packet = search.search("ἔτους", SearchFilters(document_id="dclp:DCLP/23/23944.xml"))

        assert packet.items
        assert {item.document_id for item in packet.items} == {"dclp:DCLP/23/23944.xml"}


class TestMetadataOnlyResults:
    def test_metadata_only_documents_surface(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        packet = search.search("14258")

        assert any(
            item.document_id == "dclp:DCLP/23/23702.xml" and item.kind is None
            for item in packet.items
        ), "metadata-only documents must surface for metadata searches"


class TestDeterminismAndConfig:
    def test_same_query_same_order(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        first = [i.passage_id for i in search.search("ἔτους").items]
        second = [i.passage_id for i in search.search("ἔτους").items]

        assert first == second

    def test_ranking_configuration_is_explicit(self) -> None:
        assert len(CorpusSearch.BM25_WEIGHTS) == 2
        assert CorpusSearch.BM25_WEIGHTS[0] > CorpusSearch.BM25_WEIGHTS[1], (
            "passage search text must outweigh titles in ranking"
        )

    def test_limit_bounds_results(self, corpus_artifact: Path) -> None:
        search = searcher(corpus_artifact)

        packet = search.search("ἔτους", limit=1)

        assert len(packet.items) <= 1
