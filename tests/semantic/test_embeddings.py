from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from papyrus_chat.semantic.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingKind,
    LocalEmbeddingEncoder,
    cosine_similarity,
    model_snapshot_path,
    normalize_embedding,
    prefixed_texts,
)


def test_default_model_is_pinned_and_portable() -> None:
    assert DEFAULT_EMBEDDING_MODEL.model_id == "intfloat/multilingual-e5-small"
    assert DEFAULT_EMBEDDING_MODEL.revision == "4a4cddf9cf6d77a61cc1c73f824ec2127773db85"


@pytest.mark.network
def test_pinned_model_revision_contains_configured_file() -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")

    info = huggingface_hub.HfApi().model_info(
        DEFAULT_EMBEDDING_MODEL.model_id,
        revision=DEFAULT_EMBEDDING_MODEL.revision,
    )
    files = {sibling.rfilename for sibling in info.siblings}

    assert DEFAULT_EMBEDDING_MODEL.model_file in files
    assert DEFAULT_EMBEDDING_MODEL.dimensions == 384
    assert DEFAULT_EMBEDDING_MODEL.model_file == "onnx/model_O4.onnx"
    assert DEFAULT_EMBEDDING_MODEL.revision


@pytest.mark.parametrize(
    ("kind", "texts", "expected"),
    [
        (
            "query",
            ("list of tax payments", "Steuer"),
            ("query: list of tax payments", "query: Steuer"),
        ),
        ("passage", ("Liste", "Steuern"), ("passage: Liste", "passage: Steuern")),
    ],
)
def test_e5_prefixes_are_applied(
    kind: EmbeddingKind, texts: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert prefixed_texts(texts, kind=kind) == expected


def test_custom_model_prefixes_are_used() -> None:
    custom = replace(DEFAULT_EMBEDDING_MODEL, query_prefix="q: ", passage_prefix="p: ")
    assert prefixed_texts(("text",), kind="query", model_spec=custom) == ("q: text",)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        prefixed_texts(("",), kind="query")


def test_normalize_embedding_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        normalize_embedding((float("nan"), 1.0), dimensions=2)


@pytest.mark.parametrize("model_file", ["/tmp/model.onnx", "../model.onnx", ""])
def test_model_snapshot_path_rejects_unsafe_or_missing_paths(
    tmp_path: Path, model_file: str
) -> None:
    spec = replace(DEFAULT_EMBEDDING_MODEL, model_file=model_file)

    with pytest.raises(ValueError, match="model file"):
        model_snapshot_path(tmp_path, spec)


def test_cosine_similarity_is_scale_invariant() -> None:
    left = normalize_embedding((3.0, 4.0), dimensions=2)
    scaled = normalize_embedding((30.0, 40.0), dimensions=2)

    assert cosine_similarity(left, scaled) == pytest.approx(1.0)


class FakeEmbeddingModel:
    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(float(len(texts)), float(len(text))) for text in texts]


def test_local_encoder_normalizes_and_returns_vectors() -> None:
    encoder = LocalEmbeddingEncoder(
        Path("/tmp/model"),
        model_spec=replace(DEFAULT_EMBEDDING_MODEL, dimensions=2),
        model_factory=lambda _path: FakeEmbeddingModel(),
    )

    vectors = encoder.encode(("Liste", "Steuern"), kind="passage")

    assert len(vectors) == 2
    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0)
    assert sum(value * value for value in vectors[1]) == pytest.approx(1.0)


def test_local_encoder_rejects_dimension_mismatch() -> None:
    class WrongDimensions:
        def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
            return [(1.0,) for _ in texts]

    encoder = LocalEmbeddingEncoder(
        Path("/tmp/model"), model_factory=lambda _path: WrongDimensions()
    )

    with pytest.raises(ValueError, match="384 dimensions"):
        encoder.encode(("Liste",), kind="passage")


def test_local_encoder_registers_each_model_contract_once(monkeypatch, tmp_path: Path) -> None:
    registrations: list[dict[str, object]] = []

    class FakeTextEmbedding:
        @classmethod
        def add_custom_model(cls, **kwargs) -> None:
            registrations.append(kwargs)

        def __init__(self, **_kwargs) -> None:
            pass

        def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
            return [(1.0, 0.0) for _ in texts]

    fake_fastembed = SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    fake_description = SimpleNamespace(
        ModelSource=lambda **kwargs: kwargs,
        PoolingType=SimpleNamespace(MEAN="mean"),
    )

    import importlib

    original_import = importlib.import_module

    def fake_import(name: str):
        if name == "fastembed":
            return fake_fastembed
        if name == "fastembed.common.model_description":
            return fake_description
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    (tmp_path / "model.onnx").write_bytes(b"fixture")
    spec = replace(
        DEFAULT_EMBEDDING_MODEL,
        model_id="test/registration",
        dimensions=2,
        model_file="model.onnx",
    )

    LocalEmbeddingEncoder(tmp_path, model_spec=spec)
    LocalEmbeddingEncoder(tmp_path, model_spec=spec)

    assert len(registrations) == 1
    assert registrations[0]["model_file"] == spec.model_file
