"""Transport-neutral, read-only corpus access for one validated artifact."""

from papyrus_chat.corpus.models import (
    CorpusDateInterval,
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusFacetResult,
    CorpusFacetValue,
    CorpusHit,
    CorpusInspection,
    CorpusInspectionResult,
    CorpusQuery,
    CorpusSearchResult,
    CorpusSubjectSuggestionSummary,
    SubjectSuggestion,
)
from papyrus_chat.corpus.service import CorpusService

__all__ = [
    "CorpusDateInterval",
    "CorpusDescription",
    "CorpusDocumentMatch",
    "CorpusFacetResult",
    "CorpusFacetValue",
    "CorpusHit",
    "CorpusInspection",
    "CorpusInspectionResult",
    "CorpusQuery",
    "CorpusSearchResult",
    "CorpusService",
    "CorpusSubjectSuggestionSummary",
    "SubjectSuggestion",
]
