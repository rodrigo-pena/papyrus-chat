"""Read-only Pydantic AI tools backed by structured corpus retrieval."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext

from papyrus_chat.artifact.records import ComponentRecord
from papyrus_chat.retrieval.evidence import targeted_snippet_for
from papyrus_chat.retrieval.structured import (
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusFacetResult,
    CorpusHit,
    CorpusInspection,
    CorpusQuery,
    CorpusSearchResult,
    FacetField,
    StructuredCorpusSearch,
)

INSPECT_EXCERPT_CHARS = 500


class CorpusInspectionResult(BaseModel):
    """Bounded inspection output for selected corpus documents."""

    model_config = ConfigDict(frozen=True)

    inspections: tuple[CorpusInspection, ...]


class CorpusHitSummary(BaseModel):
    """A distinct candidate document with located evidence and its citation URL."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    languages: tuple[str, ...]
    passage_kind: Literal["edition", "translation"] | None = None
    passage_language: str | None = None
    snippet: str | None = None
    line_reference: str | None = None
    canonical_url: str | None = None


class CorpusSearchSummary(BaseModel):
    """Complete normalized query, exact candidate count, and lean located hits."""

    model_config = ConfigDict(frozen=True)

    query: CorpusQuery
    assumptions: tuple[str, ...] = ()
    candidate_count: int
    truncated: bool
    hits: tuple[CorpusHitSummary, ...]
    group_candidate_counts: tuple[int, ...] | None = None


class CorpusHgvContext(BaseModel):
    """Summarized HGV metadata and date texts for one inspected document."""

    model_config = ConfigDict(frozen=True)

    title: str
    metadata: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    date_texts: tuple[str, ...] = ()


class CorpusExcerpt(BaseModel):
    """A located passage with a bounded excerpt and its line reference."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["edition", "translation"] | None = None
    language: str | None = None
    line_reference: str | None = None
    excerpt: str | None = None


class CorpusInspectionSummary(BaseModel):
    """Identity, citation URL, HGV context, and bounded excerpts for one document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    languages: tuple[str, ...]
    canonical_url: str | None = None
    hgv: CorpusHgvContext | None = None
    passages: tuple[CorpusExcerpt, ...] = ()


class CorpusInspectionOutcome(BaseModel):
    """Inspection summaries plus any requested ids the corpus does not contain."""

    model_config = ConfigDict(frozen=True)

    inspections: tuple[CorpusInspectionSummary, ...] = ()
    missing: tuple[str, ...] = ()


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

    def document_for_citation(self, canonical_url: str) -> CorpusDocumentMatch | None:
        return self._search.document_for_citation(canonical_url)


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


def _remember_corpus_urls(deps: CorpusToolDeps, urls: Iterable[str | None]) -> None:
    deps.known_corpus_urls.update(url for url in urls if url is not None)


def _search_summary(result: CorpusSearchResult) -> CorpusSearchSummary:
    return CorpusSearchSummary(
        query=result.query,
        assumptions=result.assumptions,
        candidate_count=result.candidate_count,
        truncated=result.truncated,
        hits=tuple(_hit_summary(hit) for hit in result.hits),
        group_candidate_counts=result.group_candidate_counts,
    )


def _hit_summary(hit: CorpusHit) -> CorpusHitSummary:
    return CorpusHitSummary(
        document_id=hit.document_id,
        title=hit.title,
        collection=hit.collection,
        languages=hit.languages,
        passage_kind=hit.passage_kind,
        passage_language=hit.passage_language,
        snippet=hit.snippet,
        line_reference=hit.line_reference,
        canonical_url=hit.canonical_url,
    )


def _inspection_summaries(
    inspections: Iterable[CorpusInspection],
    *,
    focus_terms: tuple[str, ...] = (),
    excerpt_chars: int = INSPECT_EXCERPT_CHARS,
) -> tuple[CorpusInspectionSummary, ...]:
    return tuple(
        CorpusInspectionSummary(
            document_id=inspection.document_id,
            title=inspection.title,
            collection=inspection.collection,
            languages=inspection.languages,
            canonical_url=inspection.canonical_url,
            hgv=_hgv_context(inspection.components),
            passages=tuple(
                _excerpt(passage, focus_terms=focus_terms, excerpt_chars=excerpt_chars)
                for passage in inspection.passages
            ),
        )
        for inspection in inspections
    )


def _inspection_outcome(
    inspections: tuple[CorpusInspection, ...],
    requested_ids: list[str],
    *,
    focus_terms: tuple[str, ...] = (),
    excerpt_chars: int = INSPECT_EXCERPT_CHARS,
) -> CorpusInspectionOutcome:
    found = {inspection.document_id for inspection in inspections}
    missing = tuple(dict.fromkeys(id for id in requested_ids if id not in found))
    return CorpusInspectionOutcome(
        inspections=_inspection_summaries(
            inspections,
            focus_terms=focus_terms,
            excerpt_chars=excerpt_chars,
        ),
        missing=missing,
    )


def _hgv_context(components: Iterable[ComponentRecord]) -> CorpusHgvContext | None:
    for component in components:
        if component.kind == "hgv":
            return CorpusHgvContext(
                title=component.title,
                metadata=component.metadata,
                date_texts=tuple(date.text for date in component.dates if date.text),
            )
    return None


def _excerpt(
    passage: CorpusHit,
    *,
    focus_terms: tuple[str, ...] = (),
    excerpt_chars: int = INSPECT_EXCERPT_CHARS,
) -> CorpusExcerpt:
    excerpt = (
        targeted_snippet_for(passage.passage_text, terms=focus_terms, length=excerpt_chars)
        if passage.passage_text is not None
        else None
    )
    return CorpusExcerpt(
        kind=passage.passage_kind,
        language=passage.passage_language,
        line_reference=passage.line_reference,
        excerpt=excerpt,
    )


def register_corpus_tools(agent: Agent[Any, Any]) -> None:
    """Register exactly the four read-only corpus tools on an agent."""
    agent.tool(describe_corpus)
    agent.tool(search_documents)
    agent.tool(inspect_documents)
    agent.tool(facet_documents)


__all__ = [
    "CorpusExcerpt",
    "CorpusHgvContext",
    "CorpusHitSummary",
    "CorpusInspectionOutcome",
    "CorpusInspectionResult",
    "CorpusInspectionSummary",
    "CorpusSearchSummary",
    "CorpusToolDeps",
    "CorpusToolService",
    "describe_corpus",
    "facet_documents",
    "inspect_documents",
    "register_corpus_tools",
    "search_documents",
]
