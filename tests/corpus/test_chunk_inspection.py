import pytest
from tests.retrieval.test_discovery import content_service as content_service

from papyrus_chat.corpus.projections import inspection_outcome


def test_inspection_selects_late_chunk_before_first_passages(content_service) -> None:
    document_id = "translations:Translations/3/3227-1.xml"
    chunk = content_service._connection.execute(
        "SELECT c.*, p.display_text FROM semantic_chunks c JOIN passages p USING (passage_id) "
        "WHERE c.document_id = ? ORDER BY p.sequence DESC, c.char_start DESC LIMIT 1",
        (document_id,),
    ).fetchone()
    result = content_service.inspect_documents(
        [document_id],
        chunk_ids=[chunk["chunk_id"]],
        excerpt_limit=1,
    )
    projected = inspection_outcome(result.inspections, [document_id], excerpt_chars=200)
    excerpt = projected.inspections[0].passages[0]
    assert excerpt.chunk_id == chunk["chunk_id"]
    assert excerpt.passage_id == chunk["passage_id"]
    assert excerpt.source is not None
    assert excerpt.excerpt is not None
    assert excerpt.source.path == "Translations/3/3227-1.xml"
    assert len(excerpt.excerpt) <= 200
    assert (
        excerpt.excerpt.strip("…") == chunk["display_text"][excerpt.char_start : excerpt.char_end]
    )
    assert excerpt.char_end >= chunk["char_start"]


def test_two_locations_in_same_passage_remain_separate_excerpts(content_service) -> None:
    document_id = "ddbdp:DDbDP/27/27093.xml"
    chunks = content_service._connection.execute(
        "SELECT chunk_id FROM semantic_chunks WHERE document_id = ? ORDER BY char_start",
        (document_id,),
    ).fetchall()
    selected = [chunks[0][0], chunks[-1][0]]
    result = content_service.inspect_documents([document_id], chunk_ids=selected, excerpt_limit=2)
    assert [p.chunk_id for p in result.inspections[0].passages] == selected


def test_inspection_rejects_unknown_chunks_and_wrong_document_ownership(content_service) -> None:
    chunk = content_service._connection.execute(
        "SELECT chunk_id FROM semantic_chunks LIMIT 1"
    ).fetchone()[0]
    for ids in [["missing-chunk"], [chunk]]:
        with pytest.raises(ValueError, match="chunk"):
            content_service.inspect_documents(["wrong-document"], chunk_ids=ids)


def test_inspection_without_chunk_ids_preserves_source_order(content_service) -> None:
    result = content_service.inspect_documents(
        ["translations:Translations/3/3227-1.xml"], excerpt_limit=1
    )
    assert result.inspections[0].passages[0].passage_text.startswith("Compte dressé")
    assert result.inspections[0].passages[0].chunk_id is None
