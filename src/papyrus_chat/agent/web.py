"""Opt-in provider-neutral web search for historical background context."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext

from papyrus_chat.agent.tools import CorpusToolDeps


class WebBackgroundResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    results: tuple[dict[str, str], ...]


def search_web_background(
    _ctx: RunContext[CorpusToolDeps],
    query: Annotated[
        str,
        Field(
            min_length=2,
            max_length=500,
            description=(
                "Historical or contextual fact to verify, such as reign dates in "
                "Egypt, Egyptian regnal-year mechanics, terminology, institutions, or geography. "
                "Ask for source links when citations are needed; do not search for corpus records."
            ),
        ),
    ],
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Maximum number of bounded web results to return, from 1 to 5.",
        ),
    ] = 3,
) -> WebBackgroundResult:
    """Search the web for historical or contextual background.

    Use this for externally sourced dates, chronologies, calendars, regnal systems,
    terminology, institutions, geography, and other context needed to interpret
    corpus evidence. Results provide snippets and source links as web-sourced
    background. Never use them to establish which papyri exist, what a papyrus
    contains, or any local corpus count.
    """
    try:
        import importlib

        DDGS = importlib.import_module("ddgs").DDGS
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "web background search requires the web-search dependency; install the web extra"
        ) from error
    results = DDGS().text(query, max_results=limit)
    return WebBackgroundResult(
        query=query,
        results=tuple(
            {key: str(item[key]) for key in ("title", "href", "body") if key in item and item[key]}
            for item in results
        ),
    )


# Keep the old Python names importable for callers that used the initial opt-in
# tool. Only ``search_web_background`` is registered with the agent.
WebTerminologyResult = WebBackgroundResult
search_web_terminology = search_web_background


__all__ = [
    "WebBackgroundResult",
    "WebTerminologyResult",
    "search_web_background",
    "search_web_terminology",
]
