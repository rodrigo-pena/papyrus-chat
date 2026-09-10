"""Batched content encoding and portable matrix generation."""

import logging
import struct
from collections import defaultdict
from collections.abc import Iterator, Sequence
from itertools import batched
from pathlib import Path
from typing import BinaryIO

from papyrus_chat.artifact.manifest import ContentIndexInfo
from papyrus_chat.artifact.records import (
    ComponentLinkRecord,
    ComponentRecord,
    DocumentRecord,
    PassageRecord,
    SemanticChunkRecord,
)
from papyrus_chat.artifact.schema import ArtifactWriter
from papyrus_chat.builder.content.chunks import passage_chunks
from papyrus_chat.builder.content.profiles import document_profile
from papyrus_chat.builder.semantic import SubjectEncoder
from papyrus_chat.semantic.embeddings import normalize_embedding
from papyrus_chat.semantic.tokenization import ContentTokenizer

LOGGER = logging.getLogger(__name__)
ENCODING_BATCH_SIZE = 64


def build_profile_index(
    output_dir: Path,
    *,
    documents: Sequence[DocumentRecord],
    passages: Sequence[PassageRecord],
    components: Sequence[ComponentRecord],
    links: Sequence[ComponentLinkRecord],
    writer: ArtifactWriter,
    encoder: SubjectEncoder,
    tokenizer: ContentTokenizer,
) -> ContentIndexInfo:
    by_document: dict[str, list[PassageRecord]] = defaultdict(list)
    for passage in passages:
        by_document[passage.document_id].append(passage)
    by_id = {component.component_id: component for component in components}
    document_components: dict[str, dict[str, ComponentRecord]] = defaultdict(dict)
    for component in components:
        if component.document_id:
            document_components[component.document_id][component.component_id] = component
    for link in links:
        owner = by_id[link.ddbdp_component_id]
        if owner.document_id:
            document_components[owner.document_id][link.hgv_component_id] = by_id[
                link.hgv_component_id
            ]
    profiles = (
        document_profile(
            document,
            by_document[document.document_id],
            list(document_components[document.document_id].values()),
            tokenizer,
        )
        for document in sorted(documents, key=lambda d: d.document_id)
    )
    relative = "semantic/profiles.f32"
    count = 0
    LOGGER.info("Encoding local document profiles")
    with (output_dir / relative).open("wb") as stream:
        for batch in batched(profiles, ENCODING_BATCH_SIZE):
            write_vectors(stream, [p.profile_text for p in batch], encoder)
            writer.insert_semantic_profiles(batch, first_row=count)
            count += len(batch)
            if count % 1024 == 0:
                LOGGER.info("Encoded %d document profiles", count)
    LOGGER.info("Encoded %d document profiles", count)
    return ContentIndexInfo(
        count=count,
        embeddings_file=relative,
        rows_hash=writer.semantic_content_hash("profiles"),
        preprocessing_version="profiles-v1",
    )


def write_vectors(stream: BinaryIO, texts: Sequence[str], encoder: SubjectEncoder) -> None:
    vectors = encoder.encode(texts, kind="passage")
    if len(vectors) != len(texts):
        raise ValueError("semantic encoder returned the wrong number of content vectors")
    for vector in vectors:
        normalized = normalize_embedding(vector, dimensions=encoder.model_spec.dimensions)
        stream.write(struct.pack(f"<{len(normalized)}f", *normalized))


def build_chunk_index(
    output_dir: Path,
    *,
    passages: Sequence[PassageRecord],
    writer: ArtifactWriter,
    encoder: SubjectEncoder,
    tokenizer: ContentTokenizer,
) -> ContentIndexInfo:
    def inputs() -> Iterator[tuple[SemanticChunkRecord, str]]:
        for passage in sorted(passages, key=lambda p: (p.document_id, p.sequence, p.passage_id)):
            for chunk in passage_chunks(passage, tokenizer):
                yield chunk, passage.display_text[chunk.char_start : chunk.char_end]

    relative = "semantic/chunks.f32"
    (output_dir / "semantic").mkdir(exist_ok=True)
    count = 0
    LOGGER.info("Encoding local text chunks")
    with (output_dir / relative).open("wb") as stream:
        for batch in batched(inputs(), ENCODING_BATCH_SIZE):
            records, texts = zip(*batch, strict=True)
            write_vectors(stream, texts, encoder)
            writer.insert_semantic_chunks(records, first_row=count)
            count += len(batch)
            if count % 1024 == 0:
                LOGGER.info("Encoded %d text chunks", count)
    LOGGER.info("Encoded %d text chunks", count)
    return ContentIndexInfo(
        count=count,
        embeddings_file=relative,
        rows_hash=writer.semantic_content_hash("chunks"),
        preprocessing_version="chunks-v1",
    )
