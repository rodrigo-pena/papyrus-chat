from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from papyrus_chat.semantic.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    LocalEmbeddingEncoder,
    prefixed_texts,
)


def test_default_model_is_pinned_and_portable() -> None:
    assert DEFAULT_EMBEDDING_MODEL.model_id == "intfloat/multilingual-e5-small"
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
    kind: str, texts: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert prefixed_texts(texts, kind=kind) == expected


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        prefixed_texts(("",), kind="query")


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
