"""Normalization used only for retrieval.

`search_text` and search queries both pass through `normalize_search_text`:
case folding plus NFD stripping of combining marks, so Greek matches are
diacritic- and case-insensitive. Display text is never normalized.
"""

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))
_QUERY_PREFIX = re.compile(r"[\s:_-]+")


def normalize_search_text(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return _WHITESPACE.sub(" ", stripped).strip()


def normalize_identifier_value(value: str) -> str:
    """Case-folded, whitespace-collapsed, zero-width-stripped identifier value."""
    folded = unicodedata.normalize("NFD", value.translate(_ZERO_WIDTH).casefold())
    stripped = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return _WHITESPACE.sub(" ", stripped).strip()


def normalize_identifier_query(query: str) -> tuple[str, str]:
    """Split a user query into (namespace, value); bare values get namespace ''.

    Recognizes 'TM 23944', 'tm:23944', 'TM-23944' style prefixes. Dotted
    values like 'p.tebt.1.7' are not split.
    """
    cleaned = query.translate(_ZERO_WIDTH).strip()
    parts = _QUERY_PREFIX.split(cleaned, maxsplit=1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return normalize_identifier_value(parts[0]), normalize_identifier_value(parts[1])
    return "", normalize_identifier_value(cleaned)
