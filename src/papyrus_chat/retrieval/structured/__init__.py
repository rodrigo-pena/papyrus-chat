"""Structured, distinct-document retrieval over a corpus artifact."""

from papyrus_chat.retrieval.structured.models import (
    CorpusDateInterval,
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusFacetResult,
    CorpusFacetValue,
    CorpusField,
    CorpusHit,
    CorpusInspection,
    CorpusQuery,
    CorpusSearchResult,
    FacetField,
)
from papyrus_chat.retrieval.structured.search import StructuredCorpusSearch

__all__ = [
    "CorpusDateInterval",
    "CorpusDescription",
    "CorpusDocumentMatch",
    "CorpusFacetResult",
    "CorpusFacetValue",
    "CorpusField",
    "CorpusHit",
    "CorpusInspection",
    "CorpusQuery",
    "CorpusSearchResult",
    "FacetField",
    "StructuredCorpusSearch",
]
