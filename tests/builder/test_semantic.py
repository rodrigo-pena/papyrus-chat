import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from papyrus_chat.artifact.records import (
    ComponentLinkRecord,
    ComponentRecord,
    SourceReference,
)
from papyrus_chat.builder.semantic import build_subject_index
from papyrus_chat.semantic.embeddings import EmbeddingModelSpec


class FakeEncoder:
    model_spec = EmbeddingModelSpec(
        model_id="test/model", revision="b" * 40, dimensions=2, model_file="model.onnx"
    )

    def encode(self, texts: Sequence[str], *, kind: str) -> tuple[tuple[float, ...], ...]:
        assert kind == "passage"
        return tuple((float(index + 1), 0.0) for index, _ in enumerate(texts))


class UnnormalizedEncoder(FakeEncoder):
    def encode(self, texts: Sequence[str], *, kind: str) -> tuple[tuple[float, ...], ...]:
        assert kind == "passage"
        return tuple((3.0, 4.0) for _ in texts)


def component(
    component_id: str, *, document_id: str | None, subjects: tuple[str, ...] = ()
) -> ComponentRecord:
    return ComponentRecord(
        component_id=component_id,
        document_id=document_id,
        kind="hgv" if document_id is None else "ddbdp",
        title=component_id,
        metadata={"subject": subjects} if subjects else {},
        source=SourceReference(
            repository_url="https://example.test", commit="a" * 40, path=component_id
        ),
    )


def test_subject_index_is_sorted_and_portable(tmp_path: Path) -> None:
    model_dir = tmp_path / "source-model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"model")
    output = tmp_path / "artifact"
    output.mkdir()
    components = [
        component("hgv:2", document_id=None, subjects=("Steuern", "Liste")),
        component("ddbdp:1", document_id="doc:1"),
        component("ddbdp:2", document_id="doc:2"),
    ]
    links = [
        ComponentLinkRecord(ddbdp_component_id="ddbdp:1", hgv_component_id="hgv:2"),
        ComponentLinkRecord(ddbdp_component_id="ddbdp:2", hgv_component_id="hgv:2"),
    ]

    result = build_subject_index(
        output,
        components=components,
        links=links,
        model_dir=model_dir,
        encoder=FakeEncoder(),
    )

    assert [row[1] for row in result.subject_rows] == ["Liste", "Steuern"]
    assert [row[3] for row in result.subject_rows] == [2, 2]
    lines = (output / "semantic/subjects.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["value"] for line in lines] == ["Liste", "Steuern"]
    assert (output / "semantic/model/model.onnx").read_bytes() == b"model"
    assert result.manifest.subject_count == 2
    assert result.manifest.model_file == "model.onnx"
    assert result.manifest.query_prefix == "query: "
    assert result.manifest.passage_prefix == "passage: "
    assert result.manifest.pooling == "mean"
    assert result.manifest.file_hashes["semantic/subjects.f32"].startswith("sha256:")
    assert len((output / "semantic/subjects.f32").read_bytes()) == 2 * 2 * 4


def test_subject_index_normalizes_custom_encoder_vectors(tmp_path: Path) -> None:
    model_dir = tmp_path / "source-model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"model")
    output = tmp_path / "artifact"
    output.mkdir()

    result = build_subject_index(
        output,
        components=[component("hgv:1", document_id=None, subjects=("Liste",))],
        links=[],
        model_dir=model_dir,
        encoder=UnnormalizedEncoder(),
    )

    import struct

    vector = struct.unpack("<2f", (output / "semantic/subjects.f32").read_bytes())
    assert vector == pytest.approx((0.6, 0.8))
    assert result.manifest.dimensions == 2
