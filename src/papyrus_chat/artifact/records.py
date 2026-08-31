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


class PassageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    passage_id: str
    document_id: str
    kind: Literal["edition", "translation"]
    sequence: int
    textpart: str | None = None
    line_reference: str | None = None
    display_text: str
    search_text: str
    source: SourceReference
    uncertainty: dict[str, int] = Field(default_factory=dict)
