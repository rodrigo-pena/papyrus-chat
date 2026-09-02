"""Opt-in provider-neutral terminology search for background context only."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext

from papyrus_chat.agent.tools import CorpusToolDeps


class WebTerminologyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    results: tuple[dict[str, str], ...]


def search_web_terminology(
    _ctx: RunContext[CorpusToolDeps],
    query: Annotated[str, Field(min_length=2, max_length=500)],
    limit: Annotated[int, Field(ge=1, le=5)] = 3,
) -> WebTerminologyResult:
    """Look up terminology/background; never use results as corpus evidence."""
    try:
        from ddgs import DDGS
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "web terminology search requires the web-search dependency; install the web extra"
        ) from error
    results = DDGS().text(query, max_results=limit)
    return WebTerminologyResult(
        query=query,
        results=tuple(
            {key: str(item[key]) for key in ("title", "href", "body") if key in item and item[key]}
            for item in results
        ),
    )


__all__ = ["WebTerminologyResult", "search_web_terminology"]
