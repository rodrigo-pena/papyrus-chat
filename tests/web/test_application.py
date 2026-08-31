"""Web application skeleton and startup validation (SPEC 9.1, 10)."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from papyrus_chat.chat.cli import app as chat_cli
from papyrus_chat.web.application import load_app, validate_startup

TEST_ENV = {"LLM_BASE_URL": "https://x.example/v1", "LLM_MODEL": "test-model"}

runner = CliRunner()


class TestLoadApp:
    def test_index_renders_plain_language_empty_state(self, corpus_artifact: Path) -> None:
        client = TestClient(load_app(corpus_artifact, env=TEST_ENV))

        response = client.get("/")

        assert response.status_code == 200
        html = response.text
        assert "Search the corpus" in html
        assert "vector" not in html.lower()
        assert "chunks" not in html.lower()
        assert "context window" not in html.lower()

    def test_index_uses_semantic_html(self, corpus_artifact: Path) -> None:
        client = TestClient(load_app(corpus_artifact, env=TEST_ENV))

        html = client.get("/").text

        assert '<html lang="en">' in html
        assert "<form" in html
        assert "<label" in html
        assert 'for="query"' in html

    def test_index_links_to_chat(self, corpus_artifact: Path) -> None:
        client = TestClient(load_app(corpus_artifact, env=TEST_ENV))

        html = client.get("/").text

        assert 'href="/chat"' in html
        assert "Ask a question" in html

    def test_navigation_always_links_to_chat(self, corpus_artifact: Path) -> None:
        client = TestClient(load_app(corpus_artifact, env=TEST_ENV))

        for path in ("/", "/search", "/documents/does-not-exist"):
            html = client.get(path).text
            assert 'href="/chat"' in html, f"missing chat link on {path}"

    def test_incompatible_schema_is_rejected(self, tmp_path: Path, corpus_artifact: Path) -> None:
        broken = tmp_path / "broken-artifact"
        shutil.copytree(corpus_artifact, broken)
        manifest = json.loads((broken / "manifest.json").read_text())
        manifest["artifact_schema_version"] = 2
        (broken / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(Exception, match="[Ss]chema"):
            load_app(broken)


class TestValidateStartup:
    def test_missing_files_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="manifest"):
            validate_startup(tmp_path / "nothing", env={})

    def test_missing_provider_config_names_variable(self, corpus_artifact: Path) -> None:
        with pytest.raises(Exception, match="LLM_BASE_URL"):
            validate_startup(corpus_artifact, env={})

    def test_valid_startup_passes(self, corpus_artifact: Path) -> None:
        validate_startup(
            corpus_artifact,
            env={"LLM_BASE_URL": "https://x.example/v1", "LLM_MODEL": "m"},
        )


class TestChatCli:
    def test_missing_artifact_fails_before_server(self) -> None:
        result = runner.invoke(chat_cli, ["--artifact", "/nonexistent/artifact"])

        assert result.exit_code != 0
        assert "artifact" in result.output.lower()

    def test_missing_provider_config_fails_before_server(self, corpus_artifact: Path) -> None:
        result = runner.invoke(
            chat_cli,
            ["--artifact", str(corpus_artifact), "--no-open"],
            env={},
        )

        combined = result.output + (result.stderr or "")
        assert result.exit_code != 0
        assert "LLM_BASE_URL" in combined
