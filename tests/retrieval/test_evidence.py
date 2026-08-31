"""Evidence packet structure: typed items with locators and citations (SPEC 8)."""

from pathlib import Path

from papyrus_chat.retrieval.search import CorpusSearch


def test_items_carry_display_text_kind_and_locator(corpus_artifact: Path) -> None:
    search = CorpusSearch(corpus_artifact / "corpus.sqlite")

    packet = search.search("ἔτους")

    assert packet.items
    for item in packet.items:
        assert item.document_id
        assert item.title
        assert item.collection
        assert item.passage_id
        assert item.kind in ("edition", "translation")
        assert item.display_text
        assert item.commit
        assert item.source_path
        assert item.citation_label


def test_citation_label_names_identifier_and_locator(corpus_artifact: Path) -> None:
    search = CorpusSearch(corpus_artifact / "corpus.sqlite")

    packet = search.search("ἔτους")

    item = next(i for i in packet.items if i.document_id == "dclp:DCLP/23/23944.xml")
    assert "23944" in item.citation_label
    assert "edition" in item.citation_label
    assert "DCLP/23/23944.xml" in item.citation_label


def test_snippet_is_shorter_readable_text(corpus_artifact: Path) -> None:
    search = CorpusSearch(corpus_artifact / "corpus.sqlite")

    packet = search.search("ἔτους")

    item = next(i for i in packet.items if i.display_text)
    assert item.snippet is not None
    assert item.display_text is not None
    assert len(item.snippet) <= len(item.display_text) + 1  # + ellipsis
    assert item.snippet.rstrip("…") in item.display_text
