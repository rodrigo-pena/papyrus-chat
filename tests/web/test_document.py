"""Document view: metadata, passages, provenance, citations."""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from papyrus_chat.web.application import load_app

TEST_ENV = {"LLM_BASE_URL": "https://x.example/v1", "LLM_MODEL": "test-model"}
EDITION_DOC = "dclp:DCLP/23/23944.xml"
METADATA_ONLY_DOC = "dclp:DCLP/23/23702.xml"
TRANSLATION_DOC = "translations:Translations/3/3643-1.xml"


def client_for(corpus_artifact: Path) -> TestClient:
    return TestClient(load_app(corpus_artifact, env=TEST_ENV))


def page(corpus_artifact: Path, document_id: str) -> str:
    from urllib.parse import quote

    return client_for(corpus_artifact).get(f"/documents/{quote(document_id, safe='')}").text


class TestEditionDocument:
    def test_renders_metadata_and_edition_passage(self, corpus_artifact: Path) -> None:
        html = page(corpus_artifact, EDITION_DOC)

        assert "Horoscope" in html
        assert "ἔτους" in html, "edition display text with Greek must render"
        assert "edition" in html.lower()

    def test_identifiers_with_namespaces_render(self, corpus_artifact: Path) -> None:
        html = page(corpus_artifact, EDITION_DOC)

        assert "TM" in html
        assert "23944" in html

    def test_provenance_shows_commit_path_locator(self, corpus_artifact: Path) -> None:
        html = page(corpus_artifact, EDITION_DOC)

        assert "DCLP/23/23944.xml" in html
        assert re.search(r"commit [0-9a-f]{12}", html), (
            "the resolved upstream commit must appear in provenance"
        )

    def test_copyable_citation_present(self, corpus_artifact: Path) -> None:
        html = page(corpus_artifact, EDITION_DOC)

        assert "citation" in html.lower()
        assert "TM 23944" in html

    def test_canonical_papyri_info_link_present(self, corpus_artifact: Path) -> None:
        html = page(corpus_artifact, EDITION_DOC)

        assert "papyri.info/current/23944" in html
        assert 'target="_blank"' in html


class TestTranslationDocument:
    def test_translation_labelled_as_source_translation(self, corpus_artifact: Path) -> None:
        html = page(corpus_artifact, TRANSLATION_DOC)

        assert "source translation" in html.lower()
        assert "sovereigns decree" in html


class TestMetadataOnlyDocument:
    def test_metadata_only_label_and_no_error(self, corpus_artifact: Path) -> None:
        response = client_for(corpus_artifact).get("/documents/dclp%3ADCLP%2F23%2F23702.xml")

        assert response.status_code == 200
        assert "metadata only" in response.text.lower()
        assert "Sb. 20 14258" in response.text

    def test_canonical_link_present_for_metadata_only(self, corpus_artifact: Path) -> None:
        html = client_for(corpus_artifact).get("/documents/dclp%3ADCLP%2F23%2F23702.xml").text

        assert "papyri.info/current/23702" in html


class TestRobustness:
    def test_unknown_document_is_404(self, corpus_artifact: Path) -> None:
        response = client_for(corpus_artifact).get("/documents/nonexistent")

        assert response.status_code == 404

    def test_hostile_document_id_is_404_not_error(self, corpus_artifact: Path) -> None:
        response = client_for(corpus_artifact).get("/documents/%3Cscript%3Ealert(1)%3C%2Fscript%3E")

        assert response.status_code == 404
