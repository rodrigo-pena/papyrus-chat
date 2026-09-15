"""Transport-neutral, read-only corpus access for one validated artifact."""

from papyrus_chat.corpus.models import (
    CorpusDateInterval,
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusDocumentSummary,
    CorpusFacetResult,
    CorpusFacetValue,
    CorpusHit,
    CorpusHitSummary,
    CorpusIdentifierLookupResult,
    CorpusInfo,
    CorpusInspection,
    CorpusInspectionResult,
    CorpusQuery,
    CorpusSearchResult,
    CorpusSearchSummary,
    CorpusSemanticCapability,
    CorpusSubjectSuggestionSummary,
    SubjectSuggestion,
)
from papyrus_chat.corpus.service import CorpusService
from papyrus_chat.retrieval.discovery.models import DiscoveryQuery, DiscoveryResult

__all__ = [
    "DiscoveryQuery",
    "DiscoveryResult",
    "CorpusDateInterval",
    "CorpusDescription",
    "CorpusDocumentMatch",
    "CorpusDocumentSummary",
    "CorpusFacetResult",
    "CorpusFacetValue",
    "CorpusHit",
    "CorpusHitSummary",
    "CorpusIdentifierLookupResult",
    "CorpusInfo",
    "CorpusInspection",
    "CorpusInspectionResult",
    "CorpusQuery",
    "CorpusSearchResult",
    "CorpusSearchSummary",
    "CorpusSemanticCapability",
    "CorpusService",
    "CorpusSubjectSuggestionSummary",
    "SubjectSuggestion",
]
