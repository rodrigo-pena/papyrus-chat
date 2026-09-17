"""Query and result models for structured corpus retrieval."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from papyrus_chat.artifact.records import ComponentRecord, SourceReference

CorpusField = Literal["title", "metadata", "transcription", "translation"]
FacetField = Literal["collection", "language", "subject", "material", "origin", "kind"]


def _loads_json_text(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _recover_swallowed_members(text: str, *, field: str) -> dict[str, object] | None:
    """Recover argument members a model swallowed into one field's string value.

    The malformed text leads with this field's JSON value and ends with the
    arguments object's closing brace; wrapping the text minus that brace parses
    the field's value together with the members that followed it.
    """
    if not text.rstrip().endswith("}"):
        return None
    try:
        recovered = json.loads('{"' + field + '": ' + text.rstrip()[:-1] + "}")
    except ValueError:
        return None
    return recovered if isinstance(recovered, dict) else None


class CorpusDateInterval(BaseModel):
    """Inclusive numeric bounds used to overlap linked HGV intervals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    not_before: int = Field(
        description="Inclusive lower bound as a proleptic year; BCE years are negative (e.g. -300)."
    )
    not_after: int = Field(
        description="Inclusive upper bound as a proleptic year; BCE years are negative; "
        "must be greater than or equal to not_before."
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_text(cls, value: object) -> object:
        if isinstance(value, str):
            return _loads_json_text(value)
        return value

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> "CorpusDateInterval":
        if self.not_before > self.not_after:
            raise ValueError(
                "not_before must be less than or equal to not_after (BCE years are negative)"
            )
        return self


class CorpusQuery(BaseModel):
    """A bounded query with OR semantics inside groups and AND between groups."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collections: tuple[str, ...] = Field(
        default=(),
        description="Collection identifiers to restrict the search, as reported by "
        "describe_corpus.",
    )
    term_groups: tuple[tuple[str, ...], ...] = Field(
        default=(),
        description="Up to 8 groups (AND between groups) of at most 16 alternative terms each "
        "(OR within a group); each term is at most 200 characters. Terms are prefix-matched "
        "against diacritic-folded word tokens: include inflected variants, since different "
        "stems (e.g. Greek augmented διεσ- vs unaugmented δια-) do not match. A flat list of "
        "strings is not accepted: every group must itself be a list of terms.",
    )
    subject_groups: tuple[tuple[str, ...], ...] = Field(
        default=(),
        description="Up to 8 groups of exact HGV subject labels (OR within a group, AND "
        "between groups). Use semantic suggestions to discover labels before filtering.",
    )
    fields: tuple[CorpusField, ...] = Field(
        default=("transcription", "translation", "title", "metadata"),
        description="Fields to search; a non-empty subset of title, metadata, transcription, "
        "translation.",
    )
    transcription_languages: tuple[str, ...] = Field(
        default=(),
        description="Edition language codes (e.g. grc) that a document's transcription must use.",
    )
    date_interval: CorpusDateInterval | None = Field(
        default=None,
        description="Optional inclusive interval overlapped with linked HGV date ranges; "
        "BCE years are negative and not_before must not exceed not_after. An HGV range "
        "missing one bound (e.g. 'nach 48') counts as its known bound.",
    )
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum documents returned, 1 to 100."
    )
    offset: int = Field(default=0, ge=0, description="Number of ranked documents to skip.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_text(cls, value: object) -> object:
        if isinstance(value, str):
            return _loads_json_text(value)
        if isinstance(value, dict):
            repaired: dict[str, object] = dict(value)
            known_fields = set(cls.model_fields)
            for key, item in value.items():
                if not isinstance(item, str):
                    continue
                if recovered := _recover_swallowed_members(item, field=key):
                    for name, member in recovered.items():
                        if name in known_fields:
                            repaired[name] = member
            return repaired
        return value

    @field_validator("collections", "transcription_languages", mode="before")
    @classmethod
    def normalize_values(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            decoded = _loads_json_text(value)
            value = decoded if isinstance(decoded, (list, tuple)) else [decoded]
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected a list of strings")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("values must be non-empty strings")
            cleaned = " ".join(item.split())
            if cleaned.casefold() not in {entry.casefold() for entry in normalized}:
                normalized.append(cleaned.casefold())
        return tuple(sorted(normalized))

    @field_validator("fields", mode="before")
    @classmethod
    def normalize_fields(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            decoded = _loads_json_text(value)
            value = decoded if isinstance(decoded, (list, tuple)) else [decoded]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("fields must contain at least one field")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("fields must contain non-empty strings")
            cleaned = item.strip().casefold()
            if cleaned not in normalized:
                normalized.append(cleaned)
        return tuple(normalized)

    @field_validator("term_groups", mode="before")
    @classmethod
    def normalize_term_groups(cls, value: object) -> tuple[tuple[str, ...], ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            decoded = _loads_json_text(value)
            if isinstance(decoded, (list, tuple)):
                value = decoded
        if not isinstance(value, (list, tuple)):
            raise ValueError("term_groups must be a list of term lists")
        if len(value) > 8:
            raise ValueError("at most 8 term groups are allowed")
        groups: list[tuple[str, ...]] = []
        for raw_group in value:
            if not isinstance(raw_group, (list, tuple)) or not raw_group:
                raise ValueError("each term group must contain at least one term")
            if len(raw_group) > 16:
                raise ValueError("each term group may contain at most 16 terms")
            terms: list[str] = []
            for raw_term in raw_group:
                if not isinstance(raw_term, str):
                    raise ValueError("terms must be strings")
                term = " ".join(raw_term.split())
                if not term:
                    raise ValueError("terms must be non-empty")
                if len(term) > 200:
                    raise ValueError("terms may contain at most 200 characters")
                if term.casefold() not in {entry.casefold() for entry in terms}:
                    terms.append(term)
            groups.append(tuple(terms))
        return tuple(groups)

    @field_validator("subject_groups", mode="before")
    @classmethod
    def normalize_subject_groups(cls, value: object) -> tuple[tuple[str, ...], ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            decoded = _loads_json_text(value)
            if isinstance(decoded, (list, tuple)):
                value = decoded
        if not isinstance(value, (list, tuple)):
            raise ValueError("subject_groups must be a list of subject-label lists")
        if len(value) > 8:
            raise ValueError("at most 8 subject groups are allowed")
        groups: list[tuple[str, ...]] = []
        for raw_group in value:
            if not isinstance(raw_group, (list, tuple)) or not raw_group:
                raise ValueError("each subject group must contain at least one label")
            if len(raw_group) > 32:
                raise ValueError("each subject group may contain at most 32 labels")
            labels: list[str] = []
            for raw_label in raw_group:
                if not isinstance(raw_label, str):
                    raise ValueError("subject labels must be strings")
                label = " ".join(raw_label.split())
                if not label:
                    raise ValueError("subject labels must be non-empty")
                if len(label) > 300:
                    raise ValueError("subject labels may contain at most 300 characters")
                if label.casefold() not in {entry.casefold() for entry in labels}:
                    labels.append(label)
            groups.append(tuple(labels))
        return tuple(groups)


class CorpusHit(BaseModel):
    """One distinct candidate document with an optional located passage."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    languages: tuple[str, ...]
    metadata: dict[str, str]
    passage_id: str | None = None
    passage_kind: Literal["edition", "translation"] | None = None
    passage_language: str | None = None
    passage_text: str | None = None
    snippet: str | None = None
    line_reference: str | None = None
    components: tuple[ComponentRecord, ...] = ()
    source: SourceReference
    canonical_url: str | None = None
    chunk_id: str | None = None
    chunk_char_start: int | None = None
    chunk_char_end: int | None = None


class CorpusSearchResult(BaseModel):
    """Complete normalized query, exact candidate count, and bounded hits."""

    model_config = ConfigDict(frozen=True)

    query: CorpusQuery
    assumptions: tuple[str, ...] = ()
    candidate_count: int
    truncated: bool
    hits: tuple[CorpusHit, ...]
    offset: int | None = None
    next_offset: int | None = None
    group_candidate_counts: tuple[int, ...] | None = None
    """Per-term-group candidate counts when the full conjunction matched nothing.

    Counts each group alone under the same non-term filters, so an empty result
    shows which group eliminated every document.
    """

    @property
    def normalized_query(self) -> CorpusQuery:
        """Alias useful to callers that name the returned query explicitly."""
        return self.query


class CorpusDescription(BaseModel):
    """Small inventory summary safe to expose to an agent."""

    model_config = ConfigDict(frozen=True)

    collections: tuple[str, ...]
    documents: int
    passages: int
    components: int
    languages: tuple[str, ...]
    collection_names: dict[str, str] = Field(
        default_factory=dict, description="Authoritative names keyed by collection identifier."
    )
    metadata_source_names: dict[str, str] = Field(
        default_factory=dict,
        description="Names of present auxiliary metadata sources, not searchable collections.",
    )


class CorpusInspection(BaseModel):
    """A selected document and a bounded set of located passages."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    languages: tuple[str, ...]
    metadata: dict[str, str]
    source: SourceReference
    canonical_url: str | None
    components: tuple[ComponentRecord, ...] = ()
    passages: tuple[CorpusHit, ...]


class CorpusFacetValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    count: int


class CorpusFacetResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: CorpusQuery
    field: FacetField
    values: tuple[CorpusFacetValue, ...]
    total_values: int = 0
    truncated: bool = False
    limit: int | None = None


class CorpusDocumentMatch(BaseModel):
    """A corpus document identified by its canonical citation URL."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
