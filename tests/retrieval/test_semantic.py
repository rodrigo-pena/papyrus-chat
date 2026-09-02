from pathlib import Path

from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.retrieval.semantic import SemanticSubjectSearch
from papyrus_chat.retrieval.structured import CorpusQuery
from papyrus_chat.semantic.embeddings import EmbeddingModelSpec


class FixtureEncoder:
    model_spec = EmbeddingModelSpec(
        model_id="test/model", revision="d" * 40, dimensions=2, model_file="model.onnx"
    )

    def encode(self, texts: list[str], *, kind: str) -> tuple[tuple[float, ...], ...]:
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
    suggestions = search.suggest_subject_values(
        "payments", scope=CorpusQuery(collections=["ddbdp"]), limit=3
    )
    assert suggestions
    assert suggestions[0].value == "Geld"
    assert suggestions[0].scoped_document_count == 1
    assert suggestions[0].scope_document_count == 1
    assert suggestions[0].coverage == 1.0
    assert suggestions[0].strategy == "semantic"
    search.close()
