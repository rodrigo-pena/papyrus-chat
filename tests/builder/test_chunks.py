import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from papyrus_chat.artifact.manifest import load_manifest
from papyrus_chat.artifact.records import PassageRecord, SourceReference
from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.cli import app
from papyrus_chat.builder.content.chunks import passage_chunks
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from tests.builder.test_semantic import FakeEncoder


class CharacterTokenizer:
    def offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple((index, index + 1) for index in range(len(text)))

    def count(self, text: str) -> int:
        return len(text)


def passage(text: str) -> PassageRecord:
    return PassageRecord(
        passage_id="p",
        document_id="d",
        kind="edition",
        language="grc",
        sequence=1,
        display_text=text,
        search_text=text,
        source=SourceReference(repository_url="url", commit="a" * 40, path="p.xml"),
    )


def test_chunks_cover_unicode_source_with_overlap_and_stable_locations() -> None:
    text = "ἄνθρωπος [ἀβγ] α\u0313 𐅵 " * 100
    chunks = list(passage_chunks(passage(text), CharacterTokenizer()))
    assert chunks == list(passage_chunks(passage(text), CharacterTokenizer()))
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    for before, after in zip(chunks, chunks[1:], strict=False):
        assert before.char_end - after.char_start == 64
    for chunk in chunks:
        assert 0 < chunk.char_end - chunk.char_start <= 384
        assert chunk.passage_id == "p"
        assert chunk.document_id == "d"


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_passages_produce_no_chunks(text: str) -> None:
    assert list(passage_chunks(passage(text), CharacterTokenizer())) == []


def test_short_passage_is_kept_intact() -> None:
    chunks = list(passage_chunks(passage("[λέξις]"), CharacterTokenizer()))
    assert [(c.char_start, c.char_end) for c in chunks] == [(0, 7)]


def test_content_requires_model_before_source_access() -> None:
    result = CliRunner().invoke(app, ["ddbdp", "--semantic-content", "--source", "missing"])
    assert result.exit_code == 2
    assert "requires --semantic-model-dir" in result.output


def test_chunk_build_is_local_portable_and_optional(
    tmp_path: Path,
    fixture_git_repo: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"model")
    (model / "tokenizer.json").write_text("{}")
    output = tmp_path / "artifact"
    build_artifact(
        ["ddbdp", "dclp", "translations"],
        output=output,
        source=LocalGitSource(fixture_git_repo),
        source_url="url",
        requested_ref="master",
        semantic_model_dir=model,
        semantic_encoder=FakeEncoder(),
        semantic_content=True,
        semantic_tokenizer=CharacterTokenizer(),
    )
    validate_artifact(output)
    manifest = load_manifest(output / "manifest.json")
    assert manifest.semantic_index is not None
    assert manifest.semantic_index.chunks is not None
    assert manifest.semantic_index.chunks.count > 4
    with sqlite3.connect(output / "corpus.sqlite") as db:
        assert db.execute(
            "SELECT DISTINCT kind FROM passages JOIN semantic_chunks USING (passage_id)"
        ).fetchall() == [("edition",), ("translation",)]
        assert (
            db.execute(
                "SELECT count(*) FROM passages p WHERE trim(display_text) != '' AND NOT EXISTS "
                "(SELECT 1 FROM semantic_chunks c WHERE c.passage_id = p.passage_id)"
            ).fetchone()[0]
            == 0
        )
    assert manifest.semantic_index.profiles is not None
    assert manifest.semantic_index.profiles.count == manifest.statistics.documents
    rebuilt = tmp_path / "rebuilt"
    build_artifact(
        ["translations", "dclp", "ddbdp"],
        output=rebuilt,
        source=LocalGitSource(fixture_git_repo),
        source_url="url",
        requested_ref="master",
        semantic_model_dir=model,
        semantic_encoder=FakeEncoder(),
        semantic_content=True,
        semantic_tokenizer=CharacterTokenizer(),
    )
    assert (
        load_manifest(rebuilt / "manifest.json").logical_content_hash
        == manifest.logical_content_hash
    )
