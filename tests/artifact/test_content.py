import sqlite3
import struct
from pathlib import Path

import pytest

from papyrus_chat.artifact.content import content_rows_hash
from papyrus_chat.artifact.hashing import file_sha256
from papyrus_chat.artifact.manifest import (
    ArtifactInvalid,
    ContentIndexInfo,
    load_manifest,
    save_manifest,
)
from papyrus_chat.artifact.validation import validate_artifact
from tests.artifact.test_manifest import write_semantic_artifact


def content_artifact(root: Path) -> Path:
    write_semantic_artifact(root)
    with sqlite3.connect(root / "corpus.sqlite") as db:
        db.execute(
            "INSERT INTO documents VALUES ('doc', 'dclp', 'Title', '[]', '{}',"
            " 'url', 'commit', 'path', NULL, NULL)"
        )
        db.execute(
            "INSERT INTO passages VALUES ('passage', 'doc', 'edition', 1, NULL, NULL,"
            " 'αβγ δεζ', 'αβγ δεζ', '{}', 'url', 'commit', 'path', NULL)"
        )
        db.execute("INSERT INTO semantic_chunks VALUES ('chunk', 'doc', 'passage', 0, 3, 0)")
        rows_hash = content_rows_hash(db, "chunks")
    vectors = root / "semantic/chunks.f32"
    vectors.write_bytes(struct.pack("<2f", 1, 0))
    tokenizer = root / "semantic/model/tokenizer.json"
    tokenizer.write_text("{}")
    manifest = load_manifest(root / "manifest.json")
    assert manifest.semantic_index is not None
    semantic = manifest.semantic_index.model_copy(
        update={
            "chunks": ContentIndexInfo(
                count=1,
                embeddings_file="semantic/chunks.f32",
                rows_hash=rows_hash,
                preprocessing_version="chunks-v1",
            ),
            "tokenizer_file": "tokenizer.json",
            "model_files": [*manifest.semantic_index.model_files, "semantic/model/tokenizer.json"],
            "file_hashes": {
                **manifest.semantic_index.file_hashes,
                "semantic/chunks.f32": file_sha256(vectors),
                "semantic/model/tokenizer.json": file_sha256(tokenizer),
            },
        }
    )
    save_manifest(root / "manifest.json", manifest.model_copy(update={"semantic_index": semantic}))
    return root


def test_content_artifact_validates_without_embedding_dependencies(tmp_path: Path) -> None:
    validate_artifact(content_artifact(tmp_path / "artifact"))


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE semantic_chunks SET char_end = 200",
        "UPDATE semantic_chunks SET document_id = 'other'",
        "UPDATE semantic_chunks SET vector_row = 2",
        "UPDATE semantic_chunks SET chunk_id = 'changed'",
    ],
)
def test_invalid_content_mapping_is_rejected(tmp_path: Path, mutation: str) -> None:
    root = content_artifact(tmp_path / "artifact")
    with sqlite3.connect(root / "corpus.sqlite") as db:
        db.execute(mutation)
    with pytest.raises(ArtifactInvalid, match="[Cc]ontent|[Cc]hunk"):
        validate_artifact(root)


def test_chunk_offsets_are_checked_even_with_matching_row_hash(tmp_path: Path) -> None:
    root = content_artifact(tmp_path / "artifact")
    with sqlite3.connect(root / "corpus.sqlite") as db:
        db.execute("UPDATE semantic_chunks SET char_end = 200")
        rows_hash = content_rows_hash(db, "chunks")
    manifest = load_manifest(root / "manifest.json")
    assert manifest.semantic_index is not None
    assert manifest.semantic_index.chunks is not None
    chunks = manifest.semantic_index.chunks.model_copy(update={"rows_hash": rows_hash})
    semantic = manifest.semantic_index.model_copy(update={"chunks": chunks})
    save_manifest(root / "manifest.json", manifest.model_copy(update={"semantic_index": semantic}))
    with pytest.raises(ArtifactInvalid, match="offsets"):
        validate_artifact(root)


@pytest.mark.parametrize("vector", [(float("nan"), 0), (0, 0), (2, 0)])
def test_invalid_content_vectors_are_rejected(tmp_path: Path, vector: tuple[float, float]) -> None:
    root = content_artifact(tmp_path / "artifact")
    path = root / "semantic/chunks.f32"
    path.write_bytes(struct.pack("<2f", *vector))
    manifest = load_manifest(root / "manifest.json")
    assert manifest.semantic_index is not None
    semantic = manifest.semantic_index.model_copy(
        update={
            "file_hashes": {
                **manifest.semantic_index.file_hashes,
                "semantic/chunks.f32": file_sha256(path),
            }
        }
    )
    save_manifest(root / "manifest.json", manifest.model_copy(update={"semantic_index": semantic}))
    with pytest.raises(ArtifactInvalid, match="vector|normalized"):
        validate_artifact(root)
