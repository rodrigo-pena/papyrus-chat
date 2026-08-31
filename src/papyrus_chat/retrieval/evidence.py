"""Evidence packets: typed retrieval results with locators and citations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

EvidenceKind = Literal["edition", "translation", None]


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
