"""Shared transport-neutral corpus result models and projections."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from papyrus_chat.retrieval.semantic import SubjectSuggestion
from papyrus_chat.retrieval.structured import (
    CorpusDateInterval,
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusFacetResult,
    CorpusFacetValue,
    CorpusHit,
    CorpusInspection,
    CorpusQuery,
    CorpusSearchResult,
)


class CorpusInspectionResult(BaseModel):
    """Bounded inspection output for selected corpus documents."""

    model_config = ConfigDict(frozen=True)

    inspections: tuple[CorpusInspection, ...]


class CorpusHitSummary(BaseModel):
    """A lean candidate document with located evidence and its citation URL."""

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
    """Exact candidate count and bounded lean search hits."""

    model_config = ConfigDict(frozen=True)

    query: CorpusQuery
    assumptions: tuple[str, ...] = ()
    candidate_count: int
    truncated: bool
    hits: tuple[CorpusHitSummary, ...]
    group_candidate_counts: tuple[int, ...] | None = None


class CorpusSubjectSuggestionSummary(BaseModel):
    """Bounded semantic labels with exact scope coverage for cohort planning."""

    model_config = ConfigDict(frozen=True)

    concept: str
    scope: CorpusQuery
    suggestions: tuple[SubjectSuggestion, ...]


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


__all__ = [
    "CorpusDateInterval",
    "CorpusDescription",
    "CorpusDocumentMatch",
    "CorpusExcerpt",
    "CorpusFacetResult",
    "CorpusFacetValue",
    "CorpusHit",
    "CorpusHitSummary",
    "CorpusHgvContext",
    "CorpusInspection",
    "CorpusInspectionOutcome",
    "CorpusInspectionResult",
    "CorpusInspectionSummary",
    "CorpusQuery",
    "CorpusSearchResult",
    "CorpusSearchSummary",
    "CorpusSubjectSuggestionSummary",
    "SubjectSuggestion",
]
