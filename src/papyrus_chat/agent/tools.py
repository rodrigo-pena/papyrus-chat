"""Read-only Pydantic AI tools backed by structured corpus retrieval."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext

from papyrus_chat.retrieval.structured import (
    CorpusDescription,
    CorpusFacetResult,
    CorpusInspection,
    CorpusQuery,
    CorpusSearchResult,
    FacetField,
    StructuredCorpusSearch,
)


class CorpusInspectionResult(BaseModel):
    """Bounded inspection output for selected corpus documents."""

    model_config = ConfigDict(frozen=True)

    inspections: tuple[CorpusInspection, ...]


@dataclass(frozen=True)
class CorpusToolDeps:
    service: "CorpusToolService"
    known_corpus_urls: set[str] = field(default_factory=set)


class CorpusToolService:
    """Application service used by tools and deterministic tests."""

    def __init__(self, search: StructuredCorpusSearch) -> None:
        self._search = search

    def describe_corpus(self) -> CorpusDescription:
        return self._search.describe()

    def search_documents(
        self,
        query: CorpusQuery,
        *,
        assumptions: tuple[str, ...] = (),
    ) -> CorpusSearchResult:
        return self._search.query(query, assumptions=assumptions)

    def inspect_documents(
        self,
        document_ids: list[str],
        *,
        excerpt_limit: int = 3,
    ) -> CorpusInspectionResult:
        return CorpusInspectionResult(
            inspections=self._search.inspect_documents(
                document_ids,
                excerpt_limit=excerpt_limit,
            )
        )

    def facet_documents(self, query: CorpusQuery, field: FacetField) -> CorpusFacetResult:
        return self._search.facet_documents(query, field)


def describe_corpus(ctx: RunContext[CorpusToolDeps]) -> CorpusDescription:
    """Describe available collections, counts, languages, and components."""
    return ctx.deps.service.describe_corpus()


def search_documents(ctx: RunContext[CorpusToolDeps], query: CorpusQuery) -> CorpusSearchResult:
    """Search distinct corpus documents with located evidence and provenance."""
    result = ctx.deps.service.search_documents(query)
    _remember_corpus_urls(ctx.deps, (hit.canonical_url for hit in result.hits))
    return result


def inspect_documents(
    ctx: RunContext[CorpusToolDeps],
    document_ids: Annotated[
        list[str],
        Field(
            max_length=20,
            description="Document identifiers from corpus tool results, at most 20.",
        ),
    ],
    excerpt_limit: Annotated[
        int,
        Field(ge=1, le=10, description="Located passages shown per document, from 1 to 10."),
    ] = 3,
) -> CorpusInspectionResult:
    """Inspect at most 20 selected documents and a bounded excerpt per document."""
    result = ctx.deps.service.inspect_documents(document_ids, excerpt_limit=excerpt_limit)
    _remember_corpus_urls(ctx.deps, (inspection.canonical_url for inspection in result.inspections))
    return result


def facet_documents(
    ctx: RunContext[CorpusToolDeps],
    query: CorpusQuery,
    field: Annotated[
        FacetField,
        Field(
            description=(
                "Facet for counting: collection, language (edition transcriptions), "
                "subject, material, or origin (HGV component metadata), or kind "
                "(edition or translation passages)."
            )
        ),
    ],
) -> CorpusFacetResult:
    """Count distinct candidate documents by a safe corpus facet."""
    return ctx.deps.service.facet_documents(query, field)


def _remember_corpus_urls(deps: CorpusToolDeps, urls: Iterable[str | None]) -> None:
    deps.known_corpus_urls.update(url for url in urls if url is not None)


def register_corpus_tools(agent: Agent[Any, Any]) -> None:
    """Register exactly the four read-only corpus tools on an agent."""
    agent.tool(describe_corpus)
    agent.tool(search_documents)
    agent.tool(inspect_documents)
    agent.tool(facet_documents)


__all__ = [
    "CorpusInspectionResult",
    "CorpusToolDeps",
    "CorpusToolService",
    "describe_corpus",
    "facet_documents",
    "inspect_documents",
    "register_corpus_tools",
    "search_documents",
]
