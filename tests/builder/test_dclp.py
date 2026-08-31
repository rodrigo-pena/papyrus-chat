"""Unit tests for the DCLP collection adapter (SPEC 6.3)."""

from pathlib import Path

import pytest

from papyrus_chat.builder.collections.dclp import parse_record
from papyrus_chat.builder.xml import ParseError
from papyrus_chat.textnorm import normalize_search_text

FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data"
INVALID_FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data-invalid"

REPO_URL = "https://github.com/papyri/idp.data.git"
COMMIT = "04568cb5ea3775d8113bb6e7edfd9c7168cf7e88"


def parse(name: str, root: Path = FIXTURES):
    path = root / name
    return parse_record(
        path.read_bytes(),
        collection="dclp",
        source_path=name,
        repository_url=REPO_URL,
        commit=COMMIT,
    )


class TestDclpEditionRecord:
    parsed = parse("DCLP/23/23944.xml")

    def test_document_identity_and_metadata(self) -> None:
        doc = self.parsed.document

        assert doc.collection == "dclp"
        assert "Horoscope" in doc.title
        assert "grc" in doc.languages
        assert doc.metadata.get("origPlace") == "Oxyrhynchos"
        assert doc.metadata.get("material") == "Papyrus"
        assert doc.source.commit == COMMIT
        assert doc.source.path == "DCLP/23/23944.xml"
        assert doc.canonical_url is not None
        assert doc.canonical_url.startswith("https://papyri.info/dclp/")

    def test_identifiers_from_multiple_namespaces(self) -> None:
        identifiers = {(i.namespace, i.value) for i in self.parsed.identifiers}

        assert ("dclp", "23944") in identifiers
        assert ("TM", "23944") in identifiers
        assert ("dclp-hybrid", "p.oxy;31;2555") in identifiers

    def test_edition_passage_keeps_uncertainty_signals(self) -> None:
        editions = [p for p in self.parsed.passages if p.kind == "edition"]

        assert len(editions) == 1
        text = editions[0].display_text
        assert "[" in text, "supplied text must be visibly bracketed"
        assert any("\u0370" <= ch <= "\u1fff" for ch in text), "must keep Greek"

    def test_uncertainty_counts_are_recorded(self) -> None:
        editions = [p for p in self.parsed.passages if p.kind == "edition"]

        uncertainty = editions[0].uncertainty
        assert uncertainty.get("supplied", 0) > 0
        assert uncertainty.get("unclear", 0) > 0

    def test_search_text_is_normalized_not_display(self) -> None:
        edition = self.parsed.passages[0]

        assert edition.search_text != edition.display_text
        assert edition.search_text == normalize_search_text(edition.display_text)
        assert not any(0x0300 <= ord(ch) <= 0x036F for ch in edition.search_text), (
            "search text must not contain combining marks"
        )

    def test_glyph_elements_produce_warnings(self) -> None:
        xml = (
            b'<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            b"<teiHeader><fileDesc><titleStmt><title>t</title></titleStmt></fileDesc></teiHeader>"
            b'<text><body><div type="edition" xml:space="preserve">'
            b'<ab>text <g type="haplography"/>more text</ab>'
            b"</div></body></text></TEI>"
        )

        parsed = parse_record(
            xml,
            collection="dclp",
            source_path="DCLP/23/inline-test.xml",
            repository_url=REPO_URL,
            commit=COMMIT,
        )

        assert parsed.warnings
        assert "<g>" in parsed.warnings[0]


class TestDclpMetadataOnlyRecord:
    parsed = parse("DCLP/23/23702.xml")

    def test_document_is_discoverable_without_passages(self) -> None:
        doc = self.parsed.document

        assert doc.title == "Sb. 20 14258"
        assert not self.parsed.passages

    def test_identifiers_present(self) -> None:
        identifiers = {(i.namespace, i.value) for i in self.parsed.identifiers}

        assert ("dclp", "23702") in identifiers
        assert ("LDAB", "23702") in identifiers


class TestFailurePaths:
    def test_malformed_record_names_collection_and_path(self) -> None:
        with pytest.raises(ParseError) as excinfo:
            parse(
                "DCLP/99/broken-record.xml",
                root=INVALID_FIXTURES,
            )

        message = str(excinfo.value)
        assert "dclp" in message
        assert "DCLP/99/broken-record.xml" in message


class TestNormalization:
    def test_greek_folds_case_and_strips_diacritics(self) -> None:
        assert normalize_search_text("ἔτους") == normalize_search_text("ΕΤΟΥΣ")

    def test_whitespace_is_collapsed(self) -> None:
        assert normalize_search_text("  a   b  ") == "a b"
