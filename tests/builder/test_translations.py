"""Unit tests for the Translations collection adapter (SPEC 3.1, 6.3)."""

from pathlib import Path

from papyrus_chat.builder.collections.translations import parse_record

FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data"
REPO_URL = "https://github.com/papyri/idp.data.git"
COMMIT = "04568cb5ea3775d8113bb6e7edfd9c7168cf7e88"


def parse(name: str):
    return parse_record(
        (FIXTURES / name).read_bytes(),
        collection="translations",
        source_path=name,
        repository_url=REPO_URL,
        commit=COMMIT,
    )


class TestStructuredTranslation:
    parsed = parse("Translations/3/3227-1.xml")

    def test_document_identity(self) -> None:
        doc = self.parsed.document

        assert doc.collection == "translations"
        assert "CdE 94" in doc.title
        assert doc.source.path == "Translations/3/3227-1.xml"
        assert doc.canonical_url is not None
        assert doc.canonical_url.startswith("https://papyri.info/ddbdp/")

    def test_translation_passages_with_textpart_structure(self) -> None:
        passages = self.parsed.passages

        assert len(passages) >= 1
        assert all(p.kind == "translation" for p in passages)
        assert any(p.textpart == "r.i" for p in passages), (
            "nested textpart labels must be joined as structural locators"
        )
        assert any(p.line_reference and "lines" in p.line_reference for p in passages)

    def test_identifiers_across_namespaces(self) -> None:
        identifiers = {(i.namespace, i.value) for i in self.parsed.identifiers}

        assert ("TM", "3227") in identifiers
        assert ("HGV", "3227") in identifiers
        assert any(namespace == "ddb-hybrid" for namespace, _ in identifiers)


class TestSimpleTranslation:
    parsed = parse("Translations/3/3643-1.xml")

    def test_single_translation_passage(self) -> None:
        passages = self.parsed.passages

        assert len(passages) == 1
        assert passages[0].kind == "translation"
        assert "sovereigns decree" in passages[0].display_text

    def test_identifiers_include_apis_and_ddbdp(self) -> None:
        identifiers = {(i.namespace, i.value) for i in self.parsed.identifiers}

        assert ("apisid", "berkeley.apis.463") in identifiers
        assert ("ddbdp", "p.tebt.1.7") in identifiers
        assert ("TM", "3643") in identifiers
