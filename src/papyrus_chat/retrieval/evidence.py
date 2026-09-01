"""Evidence packets: typed retrieval results with locators and citations."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from papyrus_chat.textnorm import normalize_search_text

EvidenceKind = Literal["edition", "translation", None]

_FOCUS_BREAKS = re.compile(r'[\s()":*^{}[\]\-]')


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    passage_id: str | None = None
    kind: EvidenceKind = None
    display_text: str | None = None
    snippet: str | None = None
    commit: str | None = None
    source_path: str | None = None
    locator: str | None = None
    citation_label: str
    canonical_url: str | None = None


class EvidencePacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    strategy: Literal["identifier", "full-text"]
    items: tuple[EvidenceItem, ...]

    @property
    def is_empty(self) -> bool:
        return not self.items


def snippet_for(display_text: str, length: int = 200) -> str:
    """A readable excerpt: the start of the display text, cut on a space."""
    if len(display_text) <= length:
        return display_text
    cut = display_text[:length]
    if " " in cut[length - 40 :]:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _focus_tokens(text: str) -> list[str]:
    return [
        normalized
        for token in _FOCUS_BREAKS.split(text)
        if (normalized := normalize_search_text(token))
    ]


def _focus_words(display_text: str) -> list[tuple[int, str]]:
    return [
        (match.start(), normalized)
        for match in re.finditer(r"\S+", display_text)
        if (normalized := normalize_search_text(match.group()))
    ]


def locate_focus(display_text: str, terms: tuple[str, ...]) -> int | None:
    """The display-text index where the earliest focus term occurs.

    A term matches when its diacritic-folded tokens prefix-match consecutive
    words, mirroring the quoted prefix tokens of FTS search.
    """
    words = _focus_words(display_text)
    anchors = []
    for term in terms:
        tokens = _focus_tokens(term)
        if not tokens or len(tokens) > len(words):
            continue
        for start in range(len(words) - len(tokens) + 1):
            if all(
                words[start + offset][1].startswith(token) for offset, token in enumerate(tokens)
            ):
                anchors.append(words[start][0])
                break
    return min(anchors) if anchors else None


def targeted_snippet_for(
    display_text: str,
    *,
    terms: tuple[str, ...] = (),
    length: int = 500,
) -> str:
    """A readable excerpt centered on the first focus-term occurrence.

    Places roughly a third of the window before the match and two thirds
    after it, cut on spaces; without terms or a match it falls back to the
    start of the display text.
    """
    anchor = locate_focus(display_text, terms) if terms else None
    if anchor is None:
        return snippet_for(display_text, length)
    start = max(0, anchor - length // 3)
    end = min(len(display_text), start + length)
    start = max(0, end - length)
    window = display_text[start:end]
    prefix = ""
    suffix = ""
    if start > 0:
        if " " in window[:40]:
            window = window[window.index(" ") + 1 :]
        prefix = "…"
    if end < len(display_text):
        if " " in window[-40:]:
            window = window.rsplit(" ", 1)[0]
        suffix = "…"
    return prefix + window + suffix
