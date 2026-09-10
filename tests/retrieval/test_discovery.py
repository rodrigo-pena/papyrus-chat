from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.discovery.models import DiscoveryQuery
from papyrus_chat.retrieval.discovery.ranking import fuse_rankings
from papyrus_chat.semantic.embeddings import EmbeddingModelSpec
from tests.builder.test_chunks import CharacterTokenizer


class DiscoveryEncoder:
    model_spec = EmbeddingModelSpec(
        model_id="test/model",
        revision="d" * 40,
        dimensions=2,
        model_file="model.onnx",
    )

    def encode(self, texts: Sequence[str], *, kind: str) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (1.0, 0.0) if kind == "query" or "sovereigns" in t else (0.0, 1.0) for t in texts
        )


@pytest.fixture
def content_service(tmp_path: Path, fixture_git_repo: Path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"model")
    (model / "tokenizer.json").write_text("{}")
    root = tmp_path / "artifact"
    build_artifact(
        ["ddbdp", "dclp", "translations"],
        output=root,
        source=LocalGitSource(fixture_git_repo),
        source_url="url",
        requested_ref="master",
        semantic_content=True,
        semantic_model_dir=model,
        semantic_encoder=DiscoveryEncoder(),
        semantic_tokenizer=CharacterTokenizer(),
    )
    monkeypatch.setattr("papyrus_chat.corpus.service._semantic_runtime_available", lambda: True)
    service = CorpusService.open(root, semantic_encoder=DiscoveryEncoder())
    yield service
    service.close()


def test_discovery_returns_ranked_documents_and_inspectable_chunks(content_service) -> None:
    query = DiscoveryQuery(text="judicial complaints", limit=2)
    result = content_service.discover_documents(query)
    assert result.available
    assert result.scope_document_count == 5
    assert result.indexed_document_count == 5
    assert len(result.hits) == 2
    assert result.hits[0].document_id == "translations:Translations/3/3643-1.xml"
    assert "chunks" in result.hits[0].channels
    assert result.hits[0].chunks[0].passage_kind == "translation"
    assert result.hits[0].chunks[0].source.path == "Translations/3/3643-1.xml"
    assert result.hits[0].chunks[0].char_end > result.hits[0].chunks[0].char_start
    assert all(len(hit.chunks) <= 2 for hit in result.hits)
    assert "candidate_count" not in result.model_dump()
    assert result == content_service.discover_documents(query)


def test_passage_filters_apply_before_ranking_and_disable_profiles(content_service) -> None:
    result = content_service.discover_documents(
        DiscoveryQuery(
            text="judicial complaints",
            passage_languages=("FR",),
            passage_kinds=("translation",),
        )
    )
    assert result.scope_document_count == 1
    assert result.indexed_document_count == 1
    assert len(result.hits) == 1
    assert result.hits[0].document_id == "translations:Translations/3/3227-1.xml"
    assert all(chunk.passage_language == "fr" for chunk in result.hits[0].chunks)
    assert "profiles" not in result.hits[0].channels


def test_empty_scope_remains_available_and_has_exact_zero_counts(content_service) -> None:
    result = content_service.discover_documents(
        DiscoveryQuery(text="taxes", collections=("absent",))
    )
    assert result.available
    assert result.scope_document_count == result.indexed_document_count == 0
    assert not result.hits


def test_date_and_transcription_filters_reuse_exact_scope(content_service) -> None:
    result = content_service.discover_documents(
        DiscoveryQuery(
            text="letter",
            transcription_languages=("grc",),
            date_interval={"not_before": 101, "not_after": 125},
        )
    )
    assert {hit.collection for hit in result.hits} == {"ddbdp"}
    assert result.scope_document_count == 1


def test_missing_content_is_explicit_and_exact_search_still_works(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)
    try:
        result = service.discover_documents(DiscoveryQuery(text="taxes"))
        assert not result.available
        assert result.unavailable_reason
        assert not result.hits
        info = service.get_corpus_info()
        assert not info.semantic_capability.chunks.available
        assert not info.semantic_capability.profiles.available
    finally:
        service.close()


@pytest.mark.parametrize(
    "values",
    [
        {"text": " "},
        {"text": "x" * 501},
        {"text": "x", "subject_groups": [["Geld"]]},
        {"text": "x", "term_groups": [["money"]]},
        {"text": "x", "limit": 101},
    ],
)
def test_discovery_rejects_lexical_gates_and_invalid_bounds(values) -> None:
    with pytest.raises(ValidationError):
        DiscoveryQuery.model_validate(values)


def test_fusion_deduplicates_each_channel_and_uses_stable_ties() -> None:
    identities = {"a": ("dclp", "a"), "b": ("dclp", "b")}
    ranked = fuse_rankings({"chunks": ["b", "b", "a"], "profiles": ["a", "b"]}, identities)
    assert [hit[0] for hit in ranked] == ["a", "b"]
    assert ranked[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_subject_and_content_tools_share_one_lazy_local_model(content_service, monkeypatch) -> None:
    from papyrus_chat.retrieval.structured import CorpusQuery

    created = []

    def factory(*args, **kwargs):
        encoder = DiscoveryEncoder()
        created.append(encoder)
        return encoder

    monkeypatch.setattr("papyrus_chat.semantic.embeddings.LocalEmbeddingEncoder", factory)
    service = CorpusService.open(content_service.artifact_root)
    try:
        assert not created
        service.suggest_subjects("money", scope=CorpusQuery())
        service.discover_documents(DiscoveryQuery(text="complaints"))
        assert len(created) == 1
    finally:
        service.close()


def test_missing_optional_runtime_is_not_a_zero_match(content_service, monkeypatch) -> None:
    monkeypatch.setattr("papyrus_chat.corpus.service._semantic_runtime_available", lambda: False)
    result = content_service.discover_documents(DiscoveryQuery(text="complaints"))
    assert not result.available
    assert result.scope_document_count is None
    assert "[semantic]" in result.unavailable_reason


def test_closed_discovery_releases_read_only_memory_maps(content_service) -> None:
    content_service.discover_documents(DiscoveryQuery(text="complaints"))
    store = content_service._search.discovery._vectors
    maps = tuple(store._maps.values())
    assert maps and not any(mapped.closed for mapped in maps)
    content_service.close()
    assert all(mapped.closed for mapped in maps)
