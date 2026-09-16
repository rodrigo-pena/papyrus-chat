"""Continuous, source-preserving passage windows over an immutable artifact."""

import base64
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from papyrus_chat.artifact.records import SourceReference


class PassageWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    passage_id: str
    passage_index: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    passage_length: int = Field(ge=0)
    source: SourceReference
    kind: str
    languages: tuple[str, ...] = ()
    line_reference: str | None = None


class DocumentPassagePage(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    snapshot_id: str
    canonical_url: str | None = None
    passage_count: int = Field(ge=0)
    windows: tuple[PassageWindow, ...] = ()
    next_cursor: str | None = None


class PassageCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = 1
    snapshot_id: str
    document_id: str
    passage_index: int = Field(ge=0)
    char_offset: int = Field(ge=0)

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode()


def read_passages(
    connection: sqlite3.Connection, snapshot_id: str, document_id: str, cursor: str | None
) -> DocumentPassagePage:
    position = PassageCursor(
        snapshot_id=snapshot_id, document_id=document_id, passage_index=0, char_offset=0
    )
    if cursor is not None:
        try:
            if len(cursor) > 4096:
                raise ValueError("too long")
            position = PassageCursor.model_validate_json(
                base64.b64decode(cursor, altchars=b"-_", validate=True)
            )
            if position.snapshot_id != snapshot_id or position.document_id != document_id:
                raise ValueError("wrong document or artifact")
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid passage cursor for this document and artifact") from error
    document = connection.execute(
        "SELECT canonical_url FROM documents WHERE document_id=?", (document_id,)
    ).fetchone()
    if document is None:
        raise ValueError("Document does not exist")
    count = connection.execute(
        "SELECT count(*) FROM passages WHERE document_id=?", (document_id,)
    ).fetchone()[0]
    index, start = position.passage_index, position.char_offset
    if index > count or (index == count and (start or cursor is not None)):
        raise ValueError("Invalid passage cursor position")
    windows = []
    while len(windows) < 5 and index < count:
        row = connection.execute(
            "SELECT * FROM passages WHERE document_id=? "
            "ORDER BY sequence, passage_id LIMIT 1 OFFSET ?",
            (document_id, index),
        ).fetchone()
        text = row["display_text"]
        if start > len(text) or (start == len(text) and start != 0):
            raise ValueError("Invalid passage cursor character position")
        end = min(len(text), start + 2000)
        languages = tuple(
            r[0]
            for r in connection.execute(
                "SELECT language FROM passage_languages WHERE passage_id=? ORDER BY language",
                (row["passage_id"],),
            )
        )
        windows.append(
            PassageWindow(
                passage_id=row["passage_id"],
                passage_index=index,
                text=text[start:end],
                char_start=start,
                char_end=end,
                passage_length=len(text),
                kind=row["kind"],
                languages=languages,
                line_reference=row["line_reference"],
                source=SourceReference(
                    repository_url=row["source_url"],
                    commit=row["source_commit"],
                    path=row["source_path"],
                    locator=row["locator"],
                ),
            )
        )
        if end == len(text):
            index, start = index + 1, 0
        else:
            start = end
    next_cursor = (
        PassageCursor(
            snapshot_id=snapshot_id, document_id=document_id, passage_index=index, char_offset=start
        ).encode()
        if index < count
        else None
    )
    return DocumentPassagePage(
        document_id=document_id,
        snapshot_id=snapshot_id,
        canonical_url=document["canonical_url"],
        passage_count=count,
        windows=tuple(windows),
        next_cursor=next_cursor,
    )
