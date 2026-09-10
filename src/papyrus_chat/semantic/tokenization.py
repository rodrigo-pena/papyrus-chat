"""Local tokenizer access with explicit, untruncated content-token budgets."""

import importlib
from pathlib import Path
from typing import Protocol


class ContentTokenizer(Protocol):
    def offsets(self, text: str) -> tuple[tuple[int, int], ...]: ...

    def count(self, text: str) -> int: ...


class LocalTokenizer:
    def __init__(self, model_dir: Path, filename: str = "tokenizer.json") -> None:
        tokenizers = importlib.import_module("tokenizers")
        path = model_dir / filename
        if not path.resolve().is_relative_to(model_dir.resolve()):
            raise ValueError("tokenizer must stay inside the model snapshot")
        self._tokenizer = tokenizers.Tokenizer.from_file(str(path))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

    def offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple(self._tokenizer.encode(text, add_special_tokens=False).offsets)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def bounded_prefix(text: str, budget: int, tokenizer: ContentTokenizer) -> str:
    """Return an original-text prefix, rechecked after boundary retokenization."""
    if budget <= 0:
        return ""
    offsets = tokenizer.offsets(text)
    if len(offsets) <= budget:
        return text
    end = offsets[budget - 1][1]
    while end and tokenizer.count(text[:end]) > budget:
        end -= 1
    return text[:end]
