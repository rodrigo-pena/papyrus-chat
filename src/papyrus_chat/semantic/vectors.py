"""Read-only, explicitly closed memory maps for bounded local cosine scans."""

import importlib
import mmap
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class LocalVectorStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._arrays: dict[str, Any] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._closed = False

    def scores(
        self,
        relative: str,
        *,
        count: int,
        dimensions: int,
        rows: Sequence[int],
        query: Sequence[float],
    ) -> list[float]:
        if self._closed:
            raise RuntimeError("semantic vector store is closed")
        if not rows:
            return []
        np = importlib.import_module("numpy")
        if relative not in self._arrays:
            path = self.root / relative
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("semantic vector file must stay inside the artifact")
            if path.stat().st_size != count * dimensions * 4:
                raise ValueError("semantic embedding file length does not match manifest")
            with path.open("rb") as stream:
                mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps[relative] = mapped
            self._arrays[relative] = np.ndarray((count, dimensions), dtype="<f4", buffer=mapped)
        matrix = self._arrays[relative]
        vector = np.asarray(query, dtype=np.float32)
        scores: list[float] = []
        for start in range(0, len(rows), 4096):
            selected = np.asarray(rows[start : start + 4096], dtype=np.intp)
            scores.extend((matrix[selected] @ vector).tolist())
        return scores

    def close(self) -> None:
        self._arrays.clear()
        for mapped in self._maps.values():
            mapped.close()
        self._maps.clear()
        self._closed = True
