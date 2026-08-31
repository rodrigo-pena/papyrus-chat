"""Accessibility audit: labels, headings, language attributes (SPEC 10)."""

import sys
from pathlib import Path

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from papyrus_chat.web.application import load_app

TEST_ENV = {"LLM_BASE_URL": "https://x.example/v1", "LLM_MODEL": "test-model"}

sys.path.insert(0, str(Path(__file__).parent.parent))


def soup_for(client: TestClient, path: str, **params) -> BeautifulSoup:
    response = client.get(path, params=params) if params else client.get(path)
    return BeautifulSoup(response.text, "html.parser")


def test_every_page_has_exactly_one_h1(corpus_artifact: Path) -> None:
    client = TestClient(load_app(corpus_artifact, env=TEST_ENV))
    pages = ["/", "/search?query=ἔτους", "/documents/dclp:DCLP/23/23944.xml"]

    for page in pages:
        soup = soup_for(client, page)
        h1s = soup.find_all("h1")
        assert len(h1s) == 1, f"{page} must have exactly one h1, found {len(h1s)}"


def test_every_form_control_has_an_associated_label(corpus_artifact: Path) -> None:
    client = TestClient(load_app(corpus_artifact, env=TEST_ENV))
    pages = ["/", "/search?query=ἔτους"]

    for page in pages:
        soup = soup_for(client, page)
        labelled_ids = {label.get("for") for label in soup.find_all("label")}
        for control in soup.find_all(["input", "select", "textarea"]):
            if control.get("type") in ("hidden", "submit"):
                continue
            assert control.get("id") in labelled_ids, (
                f"{page}: control {control.get('id') or control.get('name')} has no label"
            )


def test_selects_have_labels_in_filters(corpus_artifact: Path) -> None:
    client = TestClient(load_app(corpus_artifact, env=TEST_ENV))

    soup = soup_for(client, "/search", query="decree", collection="translations")

    for control in soup.find_all("select"):
        assert control.get("id"), "filter select needs an id for labelling"
        assert any(label.get("for") == control.get("id") for label in soup.find_all("label"))


def test_greek_passages_have_language_attributes(corpus_artifact: Path) -> None:
    client = TestClient(load_app(corpus_artifact, env=TEST_ENV))

    soup = soup_for(client, "/documents/dclp:DCLP/23/23944.xml")

    greek_passages = soup.select(".passage-text")
    assert greek_passages, "the edition document must show passages"
    assert any(p.get("lang") == "grc" for p in greek_passages), (
        "Greek edition text must be marked lang='grc'"
    )


def test_focus_visible_styles_exist() -> None:
    css = (
        Path(__file__).parent.parent.parent
        / "src"
        / "papyrus_chat"
        / "web"
        / "static"
        / "style.css"
    ).read_text(encoding="utf-8")

    assert ":focus-visible" in css or ":focus" in css
