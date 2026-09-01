"""Evidence packet structure: typed items with locators and citations."""

from pathlib import Path

from papyrus_chat.retrieval.evidence import locate_focus, snippet_for, targeted_snippet_for
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


def test_locate_focus_prefix_matches_diacritic_folded_words() -> None:
    text = "λόγος ἀναγραφὴς χρέος ἀπόδος"

    assert locate_focus(text, ("χρεο",)) == text.index("χρέος")


def test_locate_focus_matches_phrase_tokens_across_words() -> None:
    text = "λόγος χρέος ἀπόδος τοῦ ἀρταβάνου"

    assert locate_focus(text, ("χρέος αποδ",)) == text.index("χρέος")


def test_locate_focus_returns_earliest_match_across_terms() -> None:
    text = "λόγος χρέος λόγος ἀπόδος χρέος"

    assert locate_focus(text, ("χρέος", "λόγος")) == 0


def test_locate_focus_is_none_without_a_matching_term() -> None:
    assert locate_focus("λόγος ἀναγραφῆς", ("zzz",)) is None
    assert locate_focus("λόγος ἀναγραφῆς", ()) is None
    assert locate_focus("λόγος ἀναγραφῆς", ("   ",)) is None


def test_targeted_snippet_without_terms_starts_at_the_passage_start() -> None:
    text = "λόγος ἀναγραφὴς οἴνου " * 100

    assert targeted_snippet_for(text, length=200) == snippet_for(text, 200)


def test_targeted_snippet_unmatched_term_falls_back_to_the_start() -> None:
    text = "λόγος ἀναγραφὴς οἴνου " * 100

    assert targeted_snippet_for(text, terms=("zzz-absent",), length=200) == snippet_for(text, 200)


def test_targeted_snippet_centers_on_a_late_match() -> None:
    text = "λόγος ἀναγραφὴς οἴνου " * 200 + "χρέος ἀπόδος τοῦ ἀρταβάνου " + "καὶ ἄλλο " * 100
    assert "χρέος" not in snippet_for(text, 500)

    excerpt = targeted_snippet_for(text, terms=("χρεο",), length=500)

    assert "χρέος" in excerpt
    assert excerpt.startswith("…")
    assert excerpt.endswith("…")
    assert len(excerpt) <= 502


def test_targeted_snippet_keeps_short_matched_text_whole() -> None:
    text = "ἀργύριον τὸ δοσόν"

    assert targeted_snippet_for(text, terms=("ἀργυ",), length=500) == text


def test_targeted_snippet_reaches_near_the_end_when_the_match_is_late() -> None:
    text = "λόγος ἀναγραφὴς οἴνου " * 200 + "τελευτᾷ τὸ πρᾶγμα"

    excerpt = targeted_snippet_for(text, terms=("τελευτα", "τελευτᾷ"), length=300)

    assert "τελευτᾷ τὸ πρᾶγμα" in excerpt
    assert not excerpt.endswith("…")
