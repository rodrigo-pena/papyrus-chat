"""Shared lean projections for corpus-aware host integrations."""

from collections.abc import Iterable

from papyrus_chat.artifact.records import ComponentRecord
from papyrus_chat.corpus.models import (
    CorpusExcerpt,
    CorpusHgvContext,
    CorpusHit,
    CorpusHitSummary,
    CorpusInspection,
    CorpusInspectionOutcome,
    CorpusInspectionSummary,
    CorpusSearchResult,
    CorpusSearchSummary,
)
from papyrus_chat.retrieval.evidence import targeted_snippet_for

INSPECT_EXCERPT_CHARS = 500


def search_summary(result: CorpusSearchResult) -> CorpusSearchSummary:
    """Project a complete search result to the lean host-facing shape."""
    return CorpusSearchSummary(
        query=result.query,
        assumptions=result.assumptions,
        candidate_count=result.candidate_count,
        truncated=result.truncated,
        hits=tuple(hit_summary(hit) for hit in result.hits),
        group_candidate_counts=result.group_candidate_counts,
    )


def hit_summary(hit: CorpusHit) -> CorpusHitSummary:
    """Project one full hit to identity, location, and citation fields."""
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


def inspection_summaries(
    inspections: Iterable[CorpusInspection],
    *,
    focus_terms: tuple[str, ...] = (),
    excerpt_chars: int = INSPECT_EXCERPT_CHARS,
) -> tuple[CorpusInspectionSummary, ...]:
    """Project full inspections to bounded excerpts and HGV context."""
    return tuple(
        CorpusInspectionSummary(
            document_id=inspection.document_id,
            title=inspection.title,
            collection=inspection.collection,
            languages=inspection.languages,
            canonical_url=inspection.canonical_url,
            hgv=hgv_context(inspection.components),
            passages=tuple(
                excerpt(passage, focus_terms=focus_terms, excerpt_chars=excerpt_chars)
                for passage in inspection.passages
            ),
        )
        for inspection in inspections
    )


def inspection_outcome(
    inspections: tuple[CorpusInspection, ...],
    requested_ids: Iterable[str],
    *,
    focus_terms: tuple[str, ...] = (),
    excerpt_chars: int = INSPECT_EXCERPT_CHARS,
) -> CorpusInspectionOutcome:
    """Project inspections while retaining requested IDs that were not found."""
    requested = tuple(requested_ids)
    found = {inspection.document_id for inspection in inspections}
    missing = tuple(
        dict.fromkeys(document_id for document_id in requested if document_id not in found)
    )
    return CorpusInspectionOutcome(
        inspections=inspection_summaries(
            inspections,
            focus_terms=focus_terms,
            excerpt_chars=excerpt_chars,
        ),
        missing=missing,
    )


def hgv_context(components: Iterable[ComponentRecord]) -> CorpusHgvContext | None:
    """Project the first linked HGV component to metadata and date text."""
    for component in components:
        if component.kind == "hgv":
            return CorpusHgvContext(
                title=component.title,
                metadata=component.metadata,
                date_texts=tuple(date.text for date in component.dates if date.text),
            )
    return None


def excerpt(
    passage: CorpusHit,
    *,
    focus_terms: tuple[str, ...] = (),
    excerpt_chars: int = INSPECT_EXCERPT_CHARS,
) -> CorpusExcerpt:
    """Create a bounded excerpt centered on any requested focus terms."""
    excerpt_text = (
        targeted_snippet_for(passage.passage_text, terms=focus_terms, length=excerpt_chars)
        if passage.passage_text is not None
        else None
    )
    return CorpusExcerpt(
        kind=passage.passage_kind,
        language=passage.passage_language,
        line_reference=passage.line_reference,
        excerpt=excerpt_text,
    )


# Private compatibility names used by existing agent tests and integrations.
_search_summary = search_summary
_hit_summary = hit_summary
_inspection_summaries = inspection_summaries
_inspection_outcome = inspection_outcome
_hgv_context = hgv_context
_excerpt = excerpt


__all__ = [
    "INSPECT_EXCERPT_CHARS",
    "excerpt",
    "hit_summary",
    "hgv_context",
    "inspection_outcome",
    "inspection_summaries",
    "search_summary",
]
