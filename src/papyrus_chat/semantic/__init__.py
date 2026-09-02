"""Local semantic indexing helpers."""

from papyrus_chat.semantic.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModelSpec,
    LocalEmbeddingEncoder,
    prefixed_texts,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingModelSpec",
    "LocalEmbeddingEncoder",
    "prefixed_texts",
]
