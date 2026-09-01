import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from papyrus_chat.artifact.records import (
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
    SourceReference,
)
from papyrus_chat.artifact.schema import (
    ArtifactReader,
    ArtifactWriter,
    FTS5Unavailable,
    derive_document_id,
    derive_passage_id,
)


def source(path: str = "DCLP/23/23944.xml") -> SourceReference:
    return SourceReference(
        repository_url="https://github.com/papyri/idp.data.git",
        commit="0" * 40,
        path=path,
    )


def document(doc_id: str = "dclp:DCLP/23/23944.xml") -> DocumentRecord:
    return DocumentRecord(
        document_id=doc_id,
        collection="dclp",
        title="P.Oxy. 31 2555",
        languages=["grc"],
        metadata={"invNo": "Bodl. Libr. MS. Gr. class. b. 1 (P)"},
        source=source(),
        canonical_url="https://papyri.info/dclp/23944",
    )


def passage(doc_id: str = "dclp:DCLP/23/23944.xml") -> PassageRecord:
    return PassageRecord(
        passage_id=derive_passage_id(doc_id, kind="edition", sequence=1, locator="line 1"),
        document_id=doc_id,
        kind="edition",
        sequence=1,
        textpart="column i",
        line_reference="line 1",
        display_text="ἔτους ἕκτου",
        search_text="ετουσ εκτου",
        source=source(),
        uncertainty={"supplied": 2, "unclear": 3},
    )


def identifier(doc_id: str = "dclp:DCLP/23/23944.xml") -> IdentifierRecord:
    return IdentifierRecord(document_id=doc_id, namespace="TM", value="23944")


class TestRecords:
    def test_records_are_frozen(self) -> None:
        with pytest.raises(ValidationError):
            document().model_copy().__setattr__("title", "x")  # type: ignore[union-attr]

    def test_passage_kind_is_limited(self) -> None:
        data = passage().model_dump()
        data["kind"] = "commentary"

        with pytest.raises(ValidationError):
            PassageRecord.model_validate(data)


class TestStableIds:
    def test_document_id_is_deterministic(self) -> None:
        assert derive_document_id("dclp", "DCLP/23/23944.xml") == (
            derive_document_id("dclp", "DCLP/23/23944.xml")
        )

    def test_document_id_depends_on_collection_and_path(self) -> None:
        assert derive_document_id("dclp", "DCLP/23/23944.xml") != derive_document_id(
            "translations", "DCLP/23/23944.xml"
        )
        assert derive_document_id("dclp", "DCLP/23/23945.xml") != derive_document_id(
            "dclp", "DCLP/23/23944.xml"
        )

    def test_passage_id_includes_structure(self) -> None:
        first = derive_passage_id("doc", kind="edition", sequence=1, locator="line 1")
        second = derive_passage_id("doc", kind="edition", sequence=2, locator="line 2")

        assert first != second
        assert first == derive_passage_id("doc", kind="edition", sequence=1, locator="line 1")


class TestWriterReader:
    def test_missing_fts5_fails_fast(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sqlite3

        def no_fts5(conn: sqlite3.Connection) -> None:
            raise sqlite3.OperationalError("no such module: fts5")

        monkeypatch.setattr("papyrus_chat.artifact.schema._ensure_fts5", no_fts5)
        writer = ArtifactWriter(tmp_path / "corpus.sqlite")
        with pytest.raises(FTS5Unavailable, match="fts5"):
            writer.create_schema()

    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        writer = ArtifactWriter(tmp_path / "corpus.sqlite")
        writer.create_schema()
        doc, pass_, ident = document(), passage(), identifier()
        writer.insert_document(doc)
        writer.insert_passages([pass_])
        writer.insert_identifiers([ident])
        writer.commit()
        writer.close()

        reader = ArtifactReader(tmp_path / "corpus.sqlite")
        loaded_doc = reader.get_document(doc.document_id)
        assert loaded_doc == doc

        loaded_passages = reader.get_passages(doc.document_id)
        assert loaded_passages == [pass_]

        loaded_identifiers = reader.get_identifiers(doc.document_id)
        assert loaded_identifiers == [ident]

    def test_documents_can_be_inserted_and_indexed_in_bulk(self, tmp_path: Path) -> None:
        writer = ArtifactWriter(tmp_path / "corpus.sqlite")
        writer.create_schema()
        statements: list[str] = []
        writer._connection.set_trace_callback(statements.append)

        writer.insert_documents([document("doc:1"), document("doc:2")])

        assert not [
            statement
            for statement in statements
            if "SELECT title, metadata FROM documents WHERE document_id" in statement
        ]
        assert writer._connection.execute("SELECT count(*) FROM documents_fts").fetchone() == (2,)

    def test_fts_index_contains_search_text(self, tmp_path: Path) -> None:
        writer = ArtifactWriter(tmp_path / "corpus.sqlite")
        writer.create_schema()
        writer.insert_document(document())
        writer.insert_passages([passage()])
        writer.commit()
        writer.close()

        reader = ArtifactReader(tmp_path / "corpus.sqlite")
        assert reader.fts_count() == 1

    def test_orphan_passage_rejected(self, tmp_path: Path) -> None:
        writer = ArtifactWriter(tmp_path / "corpus.sqlite")
        writer.create_schema()

        orphan = passage(doc_id="dclp:missing.xml")
        with pytest.raises(sqlite3.IntegrityError):
            writer.insert_passages([orphan])

    def test_stable_ids_identical_across_write_runs(self, tmp_path: Path) -> None:
        ids = []
        for run in range(2):
            database = tmp_path / f"run{run}.sqlite"
            writer = ArtifactWriter(database)
            writer.create_schema()
            writer.insert_document(document())
            writer.insert_passages([passage()])
            writer.commit()
            writer.close()
            reader = ArtifactReader(database)
            ids.append(reader.get_passages(document().document_id)[0].passage_id)

        assert ids[0] == ids[1]
