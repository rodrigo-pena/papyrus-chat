"""Security checks for the stock Pydantic AI web integration."""

from pathlib import Path

from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from papyrus_chat.web.application import load_app


def test_provider_secret_and_legacy_content_stay_out_of_stock_ui(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    html = tmp_path / "chat.html"
    html.write_text("<main>Stock chat</main>", encoding="utf-8")
    secret = "sk-audit-secret-9876"
    app = load_app(
        corpus_artifact,
        env={
            "LLM_BASE_URL": "https://provider.example/v1",
            "LLM_MODEL": "research-model",
            "LLM_API_KEY": secret,
        },
        model=TestModel(custom_output_text="Ready."),
        html_source=html,
    )

    client = TestClient(app, base_url="http://localhost")
    response = client.get("/")

    assert response.status_code == 200
    assert secret not in response.text
    assert "search.html" not in response.text
    assert "document.html" not in response.text
