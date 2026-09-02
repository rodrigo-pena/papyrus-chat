"""Pydantic AI adapters for the transport-neutral corpus service."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import Agent, RunContext

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
from papyrus_chat.corpus.projections import (
    INSPECT_EXCERPT_CHARS,
    _excerpt,
    _hgv_context,
    _hit_summary,
    _inspection_outcome,
    _inspection_summaries,
    _search_summary,
)
from papyrus_chat.retrieval.structured import FacetField


@dataclass(frozen=True)
class CorpusToolDeps:
    service: CorpusService
    known_corpus_urls: set[str] = field(default_factory=set)


CorpusToolService = CorpusService


def describe_corpus(ctx: RunContext[CorpusToolDeps]) -> CorpusDescription:
    """Describe available collections, counts, languages, and components."""
    return ctx.deps.service.describe_corpus()


def search_documents(ctx: RunContext[CorpusToolDeps], query: CorpusQuery) -> CorpusSearchSummary:
    """Search distinct corpus documents for lean hits with located snippets and citation URLs."""
    result = ctx.deps.service.search_documents(query)
    _remember_corpus_urls(ctx.deps, (hit.canonical_url for hit in result.hits))
    return _search_summary(result)


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
) -> CorpusInspectionOutcome:
    """Inspect at most 20 selected documents with bounded excerpts and HGV context."""
    result = ctx.deps.service.inspect_documents(document_ids, excerpt_limit=excerpt_limit)
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
    "_excerpt",
    "_hgv_context",
    "_hit_summary",
    "_inspection_outcome",
    "_inspection_summaries",
    "_search_summary",
    "describe_corpus",
    "facet_documents",
    "inspect_documents",
    "register_corpus_tools",
    "search_documents",
    "suggest_subject_values",
]
