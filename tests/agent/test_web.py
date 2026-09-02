"""Provider-neutral web background tool tests."""

import sys
from types import SimpleNamespace
from typing import cast

from pydantic_ai import RunContext

from papyrus_chat.agent.tools import CorpusToolDeps
from papyrus_chat.agent.web import (
    WebBackgroundResult,
    WebTerminologyResult,
    search_web_background,
    search_web_terminology,
)


def test_search_web_background_normalizes_bounded_results(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    class FakeDDGS:
        def text(self, query: str, *, max_results: int) -> list[dict[str, str]]:
            calls.append((query, max_results))
            return [
                {
                    "title": "Claudius in Egypt",
                    "href": "https://example.test/claudius",
                    "body": "A dated historical overview.",
                    "ignored": "field",
                },
                {"title": "No body", "href": "https://example.test/second", "body": ""},
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    result = search_web_background(
        cast(RunContext[CorpusToolDeps], None),
        "Claudius reign years Egypt Egyptian regnal calendar",
        limit=2,
    )

    assert isinstance(result, WebBackgroundResult)
    assert calls == [("Claudius reign years Egypt Egyptian regnal calendar", 2)]
    assert result.results == (
        {
            "title": "Claudius in Egypt",
            "href": "https://example.test/claudius",
            "body": "A dated historical overview.",
        },
        {"title": "No body", "href": "https://example.test/second"},
    )


def test_terminology_names_remain_compatibility_aliases() -> None:
    assert search_web_terminology is search_web_background
    assert WebTerminologyResult is WebBackgroundResult
