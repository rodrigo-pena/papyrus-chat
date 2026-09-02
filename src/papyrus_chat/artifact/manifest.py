"""Manifest models for corpus artifacts."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MANIFEST_FILENAME = "manifest.json"
ARTIFACT_SCHEMA_VERSION = 3


class ArtifactInvalid(Exception):
    """Raised when a manifest or artifact directory is unusable."""


class BuilderInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str


class ManifestSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    requested_ref: str
    resolved_commit: str


class Statistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: int
    passages: int
    components: int = 0
    links: int = 0
    parse_errors: int


class SemanticIndexInfo(BaseModel):
    """Portable semantic vocabulary index bundled with a schema-v3 artifact."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    dimensions: int = Field(gt=0)
    model_file: str
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    pooling: Literal["mean"] = "mean"
    metric: Literal["cosine"] = "cosine"
    dtype: Literal["float32"] = "float32"
    subject_count: int = Field(ge=0)
    subjects_file: str
    embeddings_file: str
    model_files: list[str] = Field(default_factory=list)
    file_hashes: dict[str, str] = Field(default_factory=dict)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION
    builder: BuilderInfo
    source: ManifestSource
    collections: list[str]
    statistics: Statistics
    logical_content_hash: str
    created_at: str = Field(description="RFC 3339 timestamp")
    semantic_index: SemanticIndexInfo | None = None

    @field_validator("collections")
    @classmethod
    def collections_sorted_canonically(cls, value: list[str]) -> list[str]:
        return sorted(value)


def load_manifest(path: Path) -> ArtifactManifest:
    """Load and check a manifest file, rejecting unsupported schema majors."""
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactInvalid(f"Manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ArtifactInvalid(f"Manifest is not valid JSON: {path}: {error}") from error

    version = data.get("artifact_schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ArtifactInvalid(
            f"Manifest field artifact_schema_version must be an integer, got {version!r}"
        )
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactInvalid(
            f"Unsupported artifact schema version {version}; "
            f"this build supports schema {ARTIFACT_SCHEMA_VERSION}. "
            "Rebuild the corpus with a compatible papyrus-corpus-build."
        )

    try:
        return ArtifactManifest.model_validate(data)
    except ValidationError as error:
        raise ArtifactInvalid(
            f"Manifest does not match schema {ARTIFACT_SCHEMA_VERSION}: {error}"
        ) from error


def save_manifest(path: Path, manifest: ArtifactManifest) -> None:
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
