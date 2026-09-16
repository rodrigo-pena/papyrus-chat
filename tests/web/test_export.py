"""Downloads preserve the shareable browser transcript without running the agent."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from papyrus_chat.web.application import load_app
from papyrus_chat.web.exporting import PINNED_UI_URL


@pytest.fixture
def export_client(corpus_artifact: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    loader = AsyncMock(
        return_value=b'<!doctype html><html><body><div id="root"></div></body></html>'
    )
    monkeypatch.setattr("pydantic_ai.ui._web.app._get_ui_html", loader)
    app = load_app(
        corpus_artifact,
        env={"LLM_BASE_URL": "https://provider.example/v1", "LLM_MODEL": "research-model"},
        model=TestModel(call_tools=[], custom_output_text="Unused"),
    )
    app.state.test_html_loader = loader
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def snapshot() -> dict:
    return {
        "id": "/thread-1",
        "title": "Investigate πάπυρος",
        "messages": [
            {"id": "q1", "role": "user", "parts": [{"type": "text", "text": "πάπυρος?"}]},
            {
                "id": "a1",
                "role": "assistant",
                "metadata": {"internal": "PRIVATE_METADATA"},
                "parts": [
                    {"type": "step-start"},
                    {
                        "type": "reasoning",
                        "text": "I will inspect the corpus.",
                        "providerMetadata": {"signature": "OPAQUE_SIGNATURE"},
                    },
                    {
                        "type": "tool-search_documents",
                        "toolCallId": "search-1",
                        "state": "output-available",
                        "input": {"query": "πάπυρος"},
                        "output": {"text": "πάπυρος " * 10_000, "signature": "evidence"},
                        "callProviderMetadata": {"internal": "PRIVATE_METADATA"},
                    },
                    {
                        "type": "source-url",
                        "sourceId": "source-1",
                        "url": "https://papyri.info/ddbdp/example",
                        "title": "Source",
                    },
                    {"type": "text", "text": "Corpus **evidence** [source](https://papyri.info)."},
                ],
            },
            {"id": "q2", "role": "user", "parts": [{"type": "text", "text": "Continue."}]},
            {
                "id": "a2",
                "role": "assistant",
                "parts": [
                    {
                        "type": "dynamic-tool",
                        "toolName": "inspect_documents",
                        "toolCallId": "inspect-1",
                        "state": "output-error",
                        "input": {},
                        "errorText": "Record not found",
                    },
                    {
                        "type": "tool-read_document_passages",
                        "toolCallId": "read-1",
                        "state": "input-streaming",
                        "input": {"id": "partial"},
                    },
                    {"type": "data-research", "data": {"note": "Unfamiliar visible content"}},
                ],
            },
        ],
    }


def test_json_export_preserves_content_without_transport_metadata(export_client, snapshot):
    response = export_client.post("/api/export", json={"format": "json", "conversation": snapshot})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-disposition"].startswith('attachment; filename="papyrus-chat-')
    assert response.headers["cache-control"] == "no-store"
    document = response.json()
    assert document["schema_version"] == 1
    assert document["kind"] == "papyrus-chat-browser-conversation"
    assert document["exported_at"]
    assert document["conversation"]["id"] == snapshot["id"]
    parts = document["conversation"]["messages"][1]["parts"]
    assert parts[2]["output"] == snapshot["messages"][1]["parts"][2]["output"]
    assert parts[1] == {"type": "reasoning", "text": "I will inspect the corpus."}
    assert document["conversation"]["messages"][3] == snapshot["messages"][3]
    assert "OPAQUE_SIGNATURE" not in response.text
    assert "PRIVATE_METADATA" not in response.text
    assert "metadata" not in document["conversation"]["messages"][1]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"format": "pdf", "conversation": {}},
        {"format": "json", "conversation": {"id": "/empty", "title": "Empty", "messages": []}},
        {
            "format": "json",
            "conversation": {
                "id": "/bad",
                "title": "Bad",
                "messages": [
                    {"id": "m", "role": "assistant", "parts": [{"type": "text", "text": {}}]},
                ],
            },
        },
    ],
)
def test_export_rejects_invalid_snapshots(export_client, payload):
    assert export_client.post("/api/export", json=payload).status_code == 422


def test_export_requires_json_and_handles_malformed_json(export_client):
    assert export_client.post("/api/export", content="hello").status_code == 415
    response = export_client.post(
        "/api/export", content="{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_exports_are_independent_and_filenames_are_safe(export_client, snapshot):
    export_client.post("/api/export", json={"format": "json", "conversation": snapshot})
    snapshot["id"] = '/../../evil"\r\nheader'
    snapshot["messages"] = snapshot["messages"][:1]
    response = export_client.post("/api/export", json={"format": "json", "conversation": snapshot})
    assert len(response.json()["conversation"]["messages"]) == 1
    assert "\r" not in response.headers["content-disposition"]
    assert "../" not in response.headers["content-disposition"]


def test_export_and_assets_keep_host_protection(export_client, snapshot):
    response = export_client.post(
        "/api/export",
        json={"format": "json", "conversation": snapshot},
        headers={"host": "attacker.example"},
    )
    assert response.status_code == 421
    assert (
        export_client.get(
            "/papyrus-assets/export.js", headers={"host": "attacker.example"}
        ).status_code
        == 421
    )


def test_default_shell_pins_ui_and_serves_export_assets(export_client):
    response = export_client.get("/thread-1")
    assert response.status_code == 200
    export_client.app.state.test_html_loader.assert_awaited_once_with(PINNED_UI_URL)
    assert "@pydantic/ai-chat-ui@2.1.0/offline/" in PINNED_UI_URL
    assert "/papyrus-assets/export.js" in response.text
    for asset in ("export.js", "export-storage.js", "export.css", "shell.css"):
        assert export_client.get(f"/papyrus-assets/{asset}").status_code == 200
    assert json.loads(export_client.get("/api/health").text)
