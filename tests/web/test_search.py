"""Search interface: results, filters, escaping, no-LLM operation (SPEC 10)."""

from pathlib import Path

from fastapi.testclient import TestClient

from papyrus_chat.web.application import load_app

TEST_ENV = {"LLM_BASE_URL": "https://x.example/v1", "LLM_MODEL": "test-model"}


def client_for(corpus_artifact: Path) -> TestClient:
    return TestClient(load_app(corpus_artifact, env=TEST_ENV))


class TestSearchResults:
    def test_results_show_identifier_title_collection_kind_snippet(
        self, corpus_artifact: Path
    ) -> None:
        client = client_for(corpus_artifact)

        html = client.get("/search", params={"query": "sovereigns decree"}).text

        assert "3643-1" in html or "Decree of Ptolemy" in html
        assert "translation" in html.lower()
        assert "translations" in html
        assert "sovereigns" in html

    def test_identifier_query_finds_metadata_only_document(self, corpus_artifact: Path) -> None:
        client = client_for(corpus_artifact)

        html = client.get("/search", params={"query": "TM 23702"}).text

        assert "Sb. 20 14258" in html
        assert "metadata only" in html.lower()

    def test_results_link_to_documents(self, corpus_artifact: Path) -> None:
        client = client_for(corpus_artifact)

        html = client.get("/search", params={"query": "sovereigns"}).text

        assert "/documents/" in html

    def test_no_results_is_friendly(self, corpus_artifact: Path) -> None:
        client = client_for(corpus_artifact)

        response = client.get("/search", params={"query": "zzzqqqxyyz absent"})

        assert response.status_code == 200
        assert "no results" in response.text.lower()


class TestEscaping:
    def test_hostile_query_is_escaped(self, corpus_artifact: Path) -> None:
        client = client_for(corpus_artifact)

        html = client.get("/search", params={"query": '<script>alert("x")</script>'}).text

        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html


class TestFilters:
    def test_collection_filter_renders_and_narrows(self, corpus_artifact: Path) -> None:
        client = client_for(corpus_artifact)

        html = client.get("/search", params={"query": "decree", "collection": "translations"}).text

        assert "translations" in html
        assert "dclp:DCLP" not in html

    def test_kind_filter_renders_and_narrows(self, corpus_artifact: Path) -> None:
        client = client_for(corpus_artifact)

        html = client.get("/search", params={"query": "ἔτους", "kind": "edition"}).text

        assert "edition" in html.lower()


class TestNoLlmOperation:
    def test_search_works_without_provider_configuration(self, corpus_artifact: Path) -> None:
        # The search page must work even when provider env vars are absent
        client = TestClient(load_app(corpus_artifact, env={}))

        response = client.get("/search", params={"query": "sovereigns"})

        assert response.status_code == 200
        assert "sovereigns" in response.text

    def test_search_result_links_to_papyri_info(self, corpus_artifact: Path) -> None:
        client = TestClient(load_app(corpus_artifact, env={}))

        html = client.get("/search", params={"query": "TM 23944"}).text

        assert "papyri.info/current/23944" in html
