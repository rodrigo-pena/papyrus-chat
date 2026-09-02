import json
import sqlite3
from pathlib import Path

import pytest

from papyrus_chat.artifact.manifest import (
    MANIFEST_FILENAME,
    ArtifactManifest,
    BuilderInfo,
    ManifestSource,
    SemanticIndexInfo,
    Statistics,
    load_manifest,
    save_manifest,
)
from papyrus_chat.artifact.schema import ArtifactWriter
from papyrus_chat.artifact.validation import ArtifactInvalid, validate_artifact


def make_manifest() -> ArtifactManifest:
    return ArtifactManifest(
        builder=BuilderInfo(name="papyrus-corpus-build", version="0.2.0"),
        source=ManifestSource(
            url="https://github.com/papyri/idp.data.git",
            requested_ref="master",
            resolved_commit="0" * 40,
        ),
        collections=["dclp", "translations"],
        statistics=Statistics(documents=2, passages=3, parse_errors=0),
        logical_content_hash="sha256:abc123",
        created_at="2026-08-31T00:00:00Z",
    )


def make_semantic_index() -> SemanticIndexInfo:
    return SemanticIndexInfo(
        model_id="intfloat/multilingual-e5-small",
        revision="a" * 40,
        dimensions=384,
        model_file="onnx/model.onnx",
        subject_count=2,
        subjects_file="semantic/subjects.jsonl",
        embeddings_file="semantic/subjects.f32",
        model_files=["semantic/model/model.onnx"],
    )


def write_artifact(root: Path, manifest: ArtifactManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    save_manifest(root / MANIFEST_FILENAME, manifest)
    writer = ArtifactWriter(root / "corpus.sqlite")
    writer.create_schema()
    writer.commit()
    writer.close()
    (root / "ATTRIBUTION.md").write_text("attribution", encoding="utf-8")
    return root


class TestManifest:
    def test_round_trips_through_json(self, tmp_path: Path) -> None:
        manifest = make_manifest()
        path = tmp_path / "manifest.json"

        save_manifest(path, manifest)
        loaded = load_manifest(path)

        assert loaded == manifest

    def test_collections_sorted_canonically(self) -> None:
        unsorted_input = make_manifest().model_dump()
        unsorted_input["collections"] = ["translations", "dclp"]

        manifest = ArtifactManifest.model_validate(unsorted_input)

        assert manifest.collections == ["dclp", "translations"]

    def test_semantic_index_metadata_round_trips(self) -> None:
        manifest = make_manifest().model_copy(update={"semantic_index": make_semantic_index()})
        assert ArtifactManifest.model_validate(manifest.model_dump()) == manifest


class TestSchemaCompatibility:
    def test_unsupported_major_version_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        data = make_manifest().model_dump(mode="json")
        data["artifact_schema_version"] = 1
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ArtifactInvalid) as excinfo:
            load_manifest(path)

        message = str(excinfo.value)
        assert "3" in message
        assert "1" in message


class TestValidateArtifact:
    def test_well_formed_artifact_passes(self, tmp_path: Path) -> None:
        validate_artifact(write_artifact(tmp_path / "artifact", make_manifest()))

    def test_missing_files_are_rejected(self, tmp_path: Path) -> None:
        root = write_artifact(tmp_path / "artifact", make_manifest())
        (root / "corpus.sqlite").unlink()

        with pytest.raises(ArtifactInvalid, match="corpus.sqlite"):
            validate_artifact(root)

    def test_missing_manifest_is_rejected(self, tmp_path: Path) -> None:
        root = write_artifact(tmp_path / "artifact", make_manifest())
        (root / MANIFEST_FILENAME).unlink()

        with pytest.raises(ArtifactInvalid, match=MANIFEST_FILENAME):
            validate_artifact(root)

    def test_corrupt_sqlite_is_rejected(self, tmp_path: Path) -> None:
        root = write_artifact(tmp_path / "artifact", make_manifest())
        (root / "corpus.sqlite").write_bytes(b"definitely not a database")

        with pytest.raises(ArtifactInvalid, match="integrity"):
            validate_artifact(root)

    def test_unsupported_schema_major_is_rejected(self, tmp_path: Path) -> None:
        manifest = make_manifest().model_copy(
            update={"artifact_schema_version": 99}  # type: ignore[arg-type]
        )
        root = write_artifact(tmp_path / "artifact", manifest)

        with pytest.raises(ArtifactInvalid, match="schema"):
            validate_artifact(root)

    def test_pre_language_index_v2_artifact_requests_a_rebuild(self, tmp_path: Path) -> None:
        root = write_artifact(tmp_path / "artifact", make_manifest())
        connection = sqlite3.connect(root / "corpus.sqlite")
        connection.execute("DROP TABLE passage_languages")
        connection.commit()
        connection.close()

        with pytest.raises(ArtifactInvalid, match=r"passage_languages.*Rebuild.*0\.3\.0"):
            validate_artifact(root)

    def test_semantic_index_paths_cannot_escape_artifact(self, tmp_path: Path) -> None:
        semantic = make_semantic_index().model_copy(update={"subjects_file": "../subjects.jsonl"})
        root = write_artifact(
            tmp_path / "artifact", make_manifest().model_copy(update={"semantic_index": semantic})
        )

        with pytest.raises(ArtifactInvalid, match="stay inside"):
            validate_artifact(root)
