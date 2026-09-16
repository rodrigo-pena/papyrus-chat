"""Public, transport-neutral semantic discovery contracts."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from papyrus_chat.artifact.records import SourceReference
from papyrus_chat.retrieval.structured import CorpusDateInterval, CorpusQuery

DiscoveryChannel = Literal["profiles", "chunks", "lexical"]


class DiscoveryQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    collections: tuple[str, ...] = Field(default=(), max_length=32)
    date_interval: CorpusDateInterval | None = Field(
        default=None,
        description="Uses existing linked-HGV date-overlap semantics.",
    )
    transcription_languages: tuple[str, ...] = Field(default=(), max_length=32)
    passage_kinds: tuple[Literal["edition", "translation"], ...] = ()
    passage_languages: tuple[str, ...] = Field(
        default=(),
        max_length=32,
        description="Restrict actual matched passages, including translation-only documents.",
    )
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def unwrap_query(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("text must be non-empty")
        return cleaned

    @field_validator(
        "collections",
        "transcription_languages",
        "passage_languages",
        "passage_kinds",
        mode="before",
    )
    @classmethod
    def normalize_scope(cls, value: object) -> tuple[str, ...]:
        return CorpusQuery.normalize_values(value)


class DiscoveryChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    passage_id: str
    passage_kind: Literal["edition", "translation"]
    passage_language: str | None = None
    char_start: int
    char_end: int
    snippet: str
    line_reference: str | None = Field(
        default=None,
        description="Parent passage line reference, not a precise chunk line range.",
    )
    source: SourceReference


class DiscoveryHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    languages: tuple[str, ...]
    canonical_url: str | None = None
    score: float = Field(description="Reciprocal-rank fusion score; not a confidence probability.")
    channels: tuple[DiscoveryChannel, ...]
    channel_scores: dict[DiscoveryChannel, float]
    chunks: tuple[DiscoveryChunk, ...] = ()
    profile_snippet: str | None = Field(
        default=None,
        description="Source-derived retrieval representation, not textual evidence.",
    )
    profile_metadata_only: bool | None = None


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: DiscoveryQuery
    available: bool = True
    unavailable_reason: str | None = None
    scope_document_count: int | None = Field(
        default=None,
        description="Exact documents satisfying structural filters, not thematic matches.",
    )
    indexed_document_count: int | None = Field(
        default=None,
        description="Exact scoped documents covered by eligible content indexes.",
    )
    hits: tuple[DiscoveryHit, ...] = ()
    offset: int | None = None
    next_offset: int | None = None
    ranked_candidate_count: int | None = None
    ranked_candidates_truncated: bool = False
    channels_used: tuple[DiscoveryChannel, ...] = ()
    method: str = "Ranked candidates for inspection; no exhaustive semantic match count."
