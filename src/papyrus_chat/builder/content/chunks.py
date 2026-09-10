"""Token-bounded original-text windows with stable source offsets."""

from collections.abc import Iterator

from papyrus_chat.artifact.records import PassageRecord, SemanticChunkRecord
from papyrus_chat.semantic.tokenization import ContentTokenizer

CHUNK_TOKENS = 384
CHUNK_OVERLAP = 64


def passage_chunks(
    passage: PassageRecord,
    tokenizer: ContentTokenizer,
) -> Iterator[SemanticChunkRecord]:
    text = passage.display_text
    if not text.strip():
        return
    offsets = tokenizer.offsets(text)
    if not offsets:
        raise ValueError(f"tokenizer produced no tokens for passage {passage.passage_id}")
    token_start = 0
    char_start = 0
    while token_start < len(offsets):
        token_end = min(token_start + CHUNK_TOKENS, len(offsets))
        char_end = len(text) if token_end == len(offsets) else offsets[token_end - 1][1]
        # Subword boundaries (including multiple tokens for one Unicode character)
        # can change when a slice is encoded independently. Never rely on truncation.
        while token_end > token_start and tokenizer.count(text[char_start:char_end]) > CHUNK_TOKENS:
            token_end -= 1
            char_end = offsets[token_end - 1][1] if token_end > token_start else char_start
        if char_end <= char_start:
            raise ValueError(f"cannot fit a character in chunk budget: {passage.passage_id}")
        yield SemanticChunkRecord(
            chunk_id=f"{passage.passage_id}#chunk-v1:{char_start}:{char_end}",
            document_id=passage.document_id,
            passage_id=passage.passage_id,
            char_start=char_start,
            char_end=char_end,
        )
        if char_end == len(text):
            break
        next_token = max(token_start + 1, token_end - CHUNK_OVERLAP)
        # Ensure progress even if several subword tokens refer to the same character.
        while next_token < len(offsets) and offsets[next_token][0] <= char_start:
            next_token += 1
        if next_token >= len(offsets):
            raise ValueError(f"tokenizer offsets cannot cover passage {passage.passage_id}")
        char_start = min(char_end, offsets[next_token][0])
        token_start = next_token
