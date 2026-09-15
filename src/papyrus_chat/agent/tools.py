"""Pydantic AI adapters for the transport-neutral corpus service."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext

from papyrus_chat.agent.context.state import ResearchRunState
from papyrus_chat.corpus import (
    CorpusDescription,
    CorpusFacetResult,
    CorpusInspectionResult,
    CorpusQuery,
    CorpusService,
    CorpusSubjectSuggestionSummary,
)
from papyrus_chat.corpus.models import (
    CorpusExcerpt,
    CorpusHgvContext,
    CorpusHitSummary,
    CorpusInspectionOutcome,
    CorpusInspectionSummary,
    CorpusSearchSummary,
)
from papyrus_chat.corpus.passages import DocumentPassagePage
from papyrus_chat.corpus.projections import (
    INSPECT_EXCERPT_CHARS,
    _excerpt,
    _hgv_context,
    _hit_summary,
    _inspection_outcome,
    _inspection_summaries,
    _search_summary,
)
from papyrus_chat.retrieval.discovery.models import DiscoveryQuery, DiscoveryResult
from papyrus_chat.retrieval.structured import FacetField


@dataclass
class CorpusToolDeps:
    service: CorpusService
    known_corpus_urls: set[str] = field(default_factory=set)
    research_state: ResearchRunState = field(default_factory=ResearchRunState)


CorpusToolService = CorpusService


def describe_corpus(ctx: RunContext[CorpusToolDeps]) -> CorpusDescription:
    """Describe available collections, counts, languages, and components."""
    return ctx.deps.service.describe_corpus()


def search_documents(ctx: RunContext[CorpusToolDeps], query: CorpusQuery) -> CorpusSearchSummary:
    """Search distinct corpus documents for lean hits with located snippets and citation URLs."""
    result = ctx.deps.service.search_documents(query)
    _remember_corpus_urls(ctx.deps, (hit.canonical_url for hit in result.hits))
    return _search_summary(result)


def discover_documents(
    ctx: RunContext[CorpusToolDeps],
    query: DiscoveryQuery,
) -> DiscoveryResult:
    """Discover semantically related documents with ranked candidates for inspection."""
    result = ctx.deps.service.discover_documents(query)
    _remember_corpus_urls(ctx.deps, (hit.canonical_url for hit in result.hits))
    return result


def read_document_passages(
    ctx: RunContext[CorpusToolDeps],
    document_id: Annotated[str, Field(min_length=1, max_length=500)],
    cursor: Annotated[str | None, Field(max_length=4096)] = None,
) -> DocumentPassagePage:
    """Read up to five exact source windows. Follow next_cursor to read all text.

    Line references describe the parent passage, not a newly inferred window range.
    """
    try:
        result = ctx.deps.service.read_document_passages(document_id, cursor=cursor)
    except ValueError as error:
        raise ModelRetry(str(error)) from error
    _remember_corpus_urls(ctx.deps, [result.canonical_url])
    return result


def inspect_documents(
    ctx: RunContext[CorpusToolDeps],
    document_ids: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=20,
            description="Document identifiers from corpus tool results, from 1 to 20.",
        ),
    ],
    excerpt_limit: Annotated[
        int,
        Field(ge=1, le=10, description="Located passages shown per document, from 1 to 10."),
    ] = 3,
    excerpt_chars: Annotated[
        int,
        Field(
            ge=200,
            le=2000,
            description="Characters shown per excerpt window, from 200 to 2000.",
        ),
    ] = INSPECT_EXCERPT_CHARS,
    focus_terms: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=200)], ...],
        Field(
            max_length=8,
            description=(
                "Optional search terms or words of interest; each excerpt centers on the "
                "earliest diacritic-folded prefix match instead of the passage start. "
                "At most 8 terms, each up to 200 characters."
            ),
        ),
    ] = (),
    chunk_ids: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=300)], ...],
        Field(
            max_length=40,
            description=(
                "Optional chunk identifiers returned by discover_documents; matching "
                "chunks are shown first and centered as bounded excerpts. Chunk ids "
                "must belong to the requested documents. At most 40 identifiers."
            ),
        ),
    ] = (),
) -> CorpusInspectionOutcome:
    """Inspect at most 20 selected documents with bounded excerpts and HGV context."""
    result = ctx.deps.service.inspect_documents(
        document_ids,
        excerpt_limit=excerpt_limit,
        chunk_ids=chunk_ids,
    )
    _remember_corpus_urls(ctx.deps, (inspection.canonical_url for inspection in result.inspections))
    return _inspection_outcome(
        result.inspections,
        document_ids,
        focus_terms=focus_terms,
        excerpt_chars=excerpt_chars,
    )


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


def suggest_subject_values(
    ctx: RunContext[CorpusToolDeps],
    concept: Annotated[str, Field(min_length=1, max_length=500)],
    scope: CorpusQuery,
    limit: Annotated[int, Field(ge=1, le=30)] = 20,
) -> CorpusSubjectSuggestionSummary:
    """Suggest exact HGV subject labels for a concept within a declared scope."""
    return ctx.deps.service.suggest_subject_values(concept, scope=scope, limit=limit)


def _remember_corpus_urls(deps: CorpusToolDeps, urls: Iterable[str | None]) -> None:
    deps.known_corpus_urls.update(url for url in urls if url is not None)


def register_corpus_tools(agent: Agent[Any, Any]) -> None:
    """Register the read-only corpus tools on an agent."""
    agent.tool(describe_corpus)
    agent.tool(search_documents)
    agent.tool(inspect_documents)
    agent.tool(read_document_passages)
    agent.tool(discover_documents)
    agent.tool(facet_documents)
    agent.tool(suggest_subject_values)


__all__ = [
    "CorpusExcerpt",
    "CorpusHgvContext",
    "CorpusHitSummary",
    "CorpusInspectionOutcome",
    "CorpusInspectionResult",
    "CorpusInspectionSummary",
    "CorpusSearchSummary",
    "CorpusSubjectSuggestionSummary",
    "CorpusToolDeps",
    "CorpusToolService",
    "DiscoveryQuery",
    "DiscoveryResult",
    "_excerpt",
    "_hgv_context",
    "_hit_summary",
    "_inspection_outcome",
    "_inspection_summaries",
    "_search_summary",
    "describe_corpus",
    "discover_documents",
    "facet_documents",
    "inspect_documents",
    "register_corpus_tools",
    "search_documents",
    "suggest_subject_values",
]
