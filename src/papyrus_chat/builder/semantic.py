"""Build the portable HGV subject vocabulary and embedding files."""

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from papyrus_chat.artifact.hashing import file_sha256
from papyrus_chat.artifact.manifest import SemanticIndexInfo
from papyrus_chat.artifact.records import ComponentLinkRecord, ComponentRecord
from papyrus_chat.semantic.embeddings import EmbeddingKind, EmbeddingModelSpec
from papyrus_chat.textnorm import normalize_identifier_value


class SubjectEncoder(Protocol):
    model_spec: EmbeddingModelSpec

    def encode(
        self, texts: Sequence[str], *, kind: EmbeddingKind
    ) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True)
class SemanticIndexBuild:
    subject_rows: list[tuple[str, str, str, int]]
    manifest: SemanticIndexInfo


def build_subject_index(
    output_dir: Path,
    *,
    components: list[ComponentRecord],
    links: list[ComponentLinkRecord],
    model_dir: Path,
    encoder: SubjectEncoder,
) -> SemanticIndexBuild:
    """Write sorted subject labels, vectors, and a copied model snapshot."""
    rows = _subject_rows(components, links)
    labels = [row[1] for row in rows]
    vectors = encoder.encode(labels, kind="passage") if labels else ()
    if len(vectors) != len(rows):
        raise ValueError("semantic encoder returned the wrong number of subject vectors")
    for vector in vectors:
        if len(vector) != encoder.model_spec.dimensions:
            raise ValueError("semantic encoder returned the wrong subject dimensions")

    semantic_dir = output_dir / "semantic"
    model_output = semantic_dir / "model"
    model_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, model_output)
    subjects_path = semantic_dir / "subjects.jsonl"
    subjects_path.write_text(
        "".join(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "value": value,
                    "value_norm": value_norm,
                    "document_count": document_count,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for subject_id, value, value_norm, document_count in rows
        ),
        encoding="utf-8",
    )
    embeddings_path = semantic_dir / "subjects.f32"
    with embeddings_path.open("wb") as stream:
        for vector in vectors:
            stream.write(_pack_float32(vector))

    relative_files = [
        subjects_path.relative_to(output_dir).as_posix(),
        embeddings_path.relative_to(output_dir).as_posix(),
    ]
    relative_files.extend(
        path.relative_to(output_dir).as_posix()
        for path in sorted(model_output.rglob("*"))
        if path.is_file()
    )
    file_hashes = {relative: file_sha256(output_dir / relative) for relative in relative_files}
    return SemanticIndexBuild(
        subject_rows=rows,
        manifest=SemanticIndexInfo(
            model_id=encoder.model_spec.model_id,
            revision=encoder.model_spec.revision,
            dimensions=encoder.model_spec.dimensions,
            subject_count=len(rows),
            subjects_file=subjects_path.relative_to(output_dir).as_posix(),
            embeddings_file=embeddings_path.relative_to(output_dir).as_posix(),
            model_files=[
                path.relative_to(output_dir).as_posix()
                for path in sorted(model_output.rglob("*"))
                if path.is_file()
            ],
            file_hashes=file_hashes,
        ),
    )


def _subject_rows(
    components: list[ComponentRecord], links: list[ComponentLinkRecord]
) -> list[tuple[str, str, str, int]]:
    component_by_id = {component.component_id: component for component in components}
    documents_by_hgv: dict[str, set[str]] = {}
    for link in links:
        ddbdp = component_by_id.get(link.ddbdp_component_id)
        if ddbdp is not None and ddbdp.document_id is not None:
            documents_by_hgv.setdefault(link.hgv_component_id, set()).add(ddbdp.document_id)
    subjects: dict[str, set[str]] = {}
    for component in components:
        values = component.metadata.get("subject", ())
        if not values:
            continue
        documents = documents_by_hgv.get(component.component_id, set())
        if component.document_id is not None:
            documents = {*documents, component.document_id}
        for value in values:
            subjects.setdefault(value, set()).update(documents)
    ordered = sorted(
        subjects.items(), key=lambda item: (normalize_identifier_value(item[0]), item[0])
    )
    return [
        (f"subject-{index:06d}", value, normalize_identifier_value(value), len(documents))
        for index, (value, documents) in enumerate(ordered, start=1)
    ]


def _pack_float32(vector: tuple[float, ...]) -> bytes:
    import struct

    return struct.pack(f"<{len(vector)}f", *vector)
