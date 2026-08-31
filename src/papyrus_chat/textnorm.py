"""Normalization used only for retrieval (SPEC 6.3).

`search_text` and search queries both pass through `normalize_search_text`:
case folding plus NFD stripping of combining marks, so Greek matches are
diacritic- and case-insensitive. Display text is never normalized.
"""

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_search_text(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return _WHITESPACE.sub(" ", stripped).strip()
