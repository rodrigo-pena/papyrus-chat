"""Frozen boundary models passed between parsing, storage, and retrieval."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository_url: str
    commit: str
    path: str
    locator: str | None = None


class DocumentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    collection: str
    title: str
    languages: list[str]
    metadata: dict[str, str]
    source: SourceReference
    canonical_url: str | None = None


class IdentifierRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    namespace: str
    value: str


class ComponentIdentifierRecord(BaseModel):
    """An identifier attached to one source component."""

    model_config = ConfigDict(frozen=True)

    component_id: str
    namespace: str
    value: str


class ComponentDateRecord(BaseModel):
    """A source-preserved date interval attached to one component."""

    model_config = ConfigDict(frozen=True)

    component_id: str
    sequence: int
    not_before: str | None = None
    not_after: str | None = None
    when: str | None = None
    text: str | None = None


class ComponentRecord(BaseModel):
    """Normalized component data persisted in artifact schema v3."""

    model_config = ConfigDict(frozen=True)

    component_id: str
    document_id: str | None
    kind: str
    title: str
    languages: tuple[str, ...] = ()
    metadata: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    dates: tuple[ComponentDateRecord, ...] = ()
    identifiers: tuple[ComponentIdentifierRecord, ...] = ()
    source: SourceReference
    canonical_url: str | None = None


class ComponentLinkRecord(BaseModel):
    """A deterministic edge between two persisted source components."""

    model_config = ConfigDict(frozen=True)

    ddbdp_component_id: str
    hgv_component_id: str


class PassageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    passage_id: str
    document_id: str
    kind: Literal["edition", "translation"]
    language: str | None = None
    sequence: int
    textpart: str | None = None
    line_reference: str | None = None
    display_text: str
    search_text: str
    source: SourceReference
    uncertainty: dict[str, int] = Field(default_factory=dict)
