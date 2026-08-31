"""Integration: build → validate → serve → search → document → chat."""

import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.cli import app as build_cli
from papyrus_chat.web.application import load_app
from tests.chat.mock_provider_server import MockProviderServer
from tests.conftest import FIXTURES, make_git_repo

runner = CliRunner()


def _build_artifact(tmp_path: Path) -> Path:
    repo = make_git_repo(FIXTURES, tmp_path / "source")
    result = runner.invoke(
        build_cli,
        [
            "dclp",
            "translations",
            "--output",
            str(tmp_path / "papyrus-corpus"),
            "--source",
            str(repo),
            "--ref",
            "master",
        ],
    )
    assert result.exit_code == 0, result.output
    return repo


def test_full_journey_from_clean_source_to_cited_answer(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    artifact = tmp_path / "papyrus-corpus"

    validate_artifact(artifact)

    with MockProviderServer(
        content="According to [1], the sovereigns issue a decree about finances."
    ) as mock:
        client = TestClient(
            load_app(artifact, env={"LLM_BASE_URL": mock.base_url, "LLM_MODEL": "m"})
        )

        # 1. Search without any LLM contact
        search_html = client.get("/search", params={"query": "sovereigns decree"}).text
        assert "Decree of Ptolemy" in search_html

        # 2. Open a document
        doc_html = client.get("/documents/translations%3ATranslations%2F3%2F3643-1.xml").text
        assert "sovereigns decree" in doc_html
        assert "source translation" in doc_html.lower()

        # 3. Ask a question; the answer cites the exact evidence supplied
        chat_html = client.post("/chat", data={"query": "sovereigns decree"}).text
        assert "Evidence used" in chat_html
        assert "model-generated" in chat_html.lower()

        request_body = mock.requests[0]["body"]
        sent_messages = " ".join(m["content"] for m in request_body["messages"])
        assert "sovereigns" in sent_messages, "evidence must be sent to the model"
        user_message = next(m["content"] for m in request_body["messages"] if m["role"] == "user")

        # The marker the model used must resolve to evidence that is displayed,
        # with the same citation label that was sent to the model.
        assert re.search(r'class="marker">\[1\]<', chat_html), (
            "the answer's evidence markers must be displayed"
        )
        sent_citation = re.search(r"\[1\] (.+)", user_message)
        assert sent_citation is not None
        assert sent_citation.group(1).splitlines()[0] in chat_html


def test_artifact_survives_source_removal(tmp_path: Path) -> None:
    repo = _build_artifact(tmp_path)
    artifact = tmp_path / "papyrus-corpus"

    shutil.rmtree(repo)

    validate_artifact(artifact)

    client = TestClient(
        load_app(artifact, env={"LLM_BASE_URL": "http://127.0.0.1:9", "LLM_MODEL": "m"})
    )
    assert client.get("/search", params={"query": "sovereigns"}).status_code == 200
    assert "Decree of Ptolemy" in client.get("/search", params={"query": "sovereigns"}).text
