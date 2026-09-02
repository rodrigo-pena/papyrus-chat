from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.retrieval.semantic import SemanticSubjectSearch
from papyrus_chat.retrieval.structured import CorpusQuery
from papyrus_chat.semantic import embeddings as embeddings_module
from papyrus_chat.semantic.embeddings import EmbeddingModelSpec


class FixtureEncoder:
    model_spec = EmbeddingModelSpec(
        model_id="test/model", revision="d" * 40, dimensions=2, model_file="model.onnx"
    )

    def encode(self, texts: Sequence[str], *, kind: str) -> tuple[tuple[float, ...], ...]:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append((0.0, 1.0) if "geld" in lowered or kind == "query" else (1.0, 0.0))
        return tuple(vectors)


def test_suggestions_fuse_local_vectors_and_report_scoped_coverage(
    tmp_path: Path, fixture_git_repo: Path
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fixture model")
    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
        semantic_model_dir=model_dir,
        semantic_encoder=FixtureEncoder(),
    )

    search = SemanticSubjectSearch(artifact / "corpus.sqlite", encoder=FixtureEncoder())
    statements: list[str] = []
    search._connection.set_trace_callback(statements.append)  # noqa: SLF001 - query budget test
    suggestions = search.suggest_subject_values(
        "payments", scope=CorpusQuery(collections=["ddbdp"]), limit=3
    )
    assert suggestions
    assert suggestions[0].value == "Geld"
    assert suggestions[0].scoped_document_count == 1
    assert suggestions[0].scope_document_count == 1
    assert suggestions[0].subject_annotated_document_count == 1
    assert suggestions[0].label_prevalence == 1.0
    assert suggestions[0].subject_annotation_coverage == 1.0
    assert suggestions[0].model_dump()["label_prevalence"] == 1.0
    assert suggestions[0].strategy == "semantic"
    assert not any("subject.value IN" in statement for statement in statements)
    search.close()


def test_query_encoder_uses_the_artifact_model_contract(
    tmp_path: Path, fixture_git_repo: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "custom.onnx").write_bytes(b"fixture model")
    spec = replace(
        FixtureEncoder.model_spec,
        model_file="custom.onnx",
        query_prefix="q: ",
        passage_prefix="p: ",
    )

    class BuildEncoder(FixtureEncoder):
        model_spec = spec

    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
        semantic_model_dir=model_dir,
        semantic_encoder=BuildEncoder(),
    )

    captured: dict[str, object] = {}

    class ReopenedEncoder(FixtureEncoder):
        def __init__(self, _model_dir: Path, *, model_spec) -> None:
            captured["spec"] = model_spec

    monkeypatch.setattr(embeddings_module, "LocalEmbeddingEncoder", ReopenedEncoder)
    search = SemanticSubjectSearch(artifact / "corpus.sqlite")
    search.suggest_subject_values("payments", scope=CorpusQuery(collections=["ddbdp"]), limit=3)

    assert captured["spec"] == spec
    search.close()
