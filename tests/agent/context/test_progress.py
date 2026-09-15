"""Progress derives from exact tool coverage, not narrative claims."""

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from papyrus_chat.agent.context.evidence import EvidenceLedger


def exchange(tool, arguments, result, call_id):
    return [
        ModelResponse(parts=[ToolCallPart(tool, arguments, call_id)]),
        ModelRequest(parts=[ToolReturnPart(tool, result, call_id)]),
    ]


def search_result(ids, *, offset=0, next_offset=None):
    return {
        "query": {},
        "candidate_count": 3,
        "truncated": next_offset is not None,
        "offset": offset,
        "next_offset": next_offset,
        "hits": [
            {
                "document_id": document_id,
                "title": document_id,
                "collection": "ddbdp",
                "languages": ["grc"],
                "canonical_url": f"https://papyri.info/ddbdp/{document_id}",
            }
            for document_id in ids
        ],
    }


def test_equivalent_results_share_stable_record_identity_but_keep_executions():
    ledger = EvidenceLedger()
    result = search_result(["doc-1"], next_offset=1)
    ledger.ingest(exchange("search_documents", {"query": {}}, result, "first"))
    first_id = ledger.records[0].record_id
    ledger.ingest(exchange("search_documents", {"query": {}}, result, "second"))
    ledger.ingest(exchange("search_documents", {"query": {}}, result, "second"))

    assert len(ledger.records) == 1
    assert ledger.records[0].record_id == first_id
    assert len(ledger.executions) == 2
    reconstructed = EvidenceLedger()
    reconstructed.ingest(exchange("search_documents", {"query": {}}, result, "different"))
    assert reconstructed.records[0].record_id == first_id


def test_search_pages_track_outstanding_distinct_candidates():
    ledger = EvidenceLedger()
    ledger.ingest(
        exchange(
            "search_documents", {"query": {}}, search_result(["doc-1"], next_offset=1), "page-1"
        )
    )
    partial = ledger.progress()
    assert set(partial["candidate_documents"]) == {"doc-1"}
    assert partial["searches"][0]["outstanding_count"] == 2

    ledger.ingest(
        exchange(
            "search_documents",
            {"query": {"offset": 1}},
            search_result(["doc-2", "doc-3"], offset=1),
            "page-2",
        )
    )
    complete = ledger.progress()
    assert set(complete["candidate_documents"]) == {"doc-1", "doc-2", "doc-3"}
    assert len(complete["searches"]) == 1
    assert complete["searches"][0]["outstanding_count"] == 0


def test_excerpt_inspection_does_not_claim_full_text_coverage():
    ledger = EvidenceLedger()
    ledger.ingest(
        exchange(
            "search_documents", {"query": {}}, search_result(["doc-1"], next_offset=1), "search"
        )
    )
    ledger.ingest(
        exchange(
            "inspect_documents",
            {"document_ids": ["doc-1"]},
            {
                "inspections": [
                    {
                        "document_id": "doc-1",
                        "title": "Document",
                        "collection": "ddbdp",
                        "languages": ["grc"],
                        "canonical_url": "https://papyri.info/ddbdp/doc-1",
                        "passages": [
                            {
                                "kind": "edition",
                                "language": "grc",
                                "line_reference": "1-2",
                                "excerpt": "πάπυρος",
                            }
                        ],
                    }
                ],
                "missing": [],
            },
            "inspect",
        )
    )

    progress = ledger.progress()
    assert set(progress["candidate_documents"]) == {"doc-1"}
    assert set(progress["excerpt_documents"]) == {"doc-1"}
    assert not progress["full_text_documents"]


def test_legacy_search_without_pagination_metadata_does_not_claim_complete_coverage():
    result = search_result(["doc-1"])
    del result["offset"]
    del result["next_offset"]
    result["truncated"] = False
    ledger = EvidenceLedger()
    ledger.ingest(exchange("search_documents", {"query": {}}, result, "legacy"))

    progress = ledger.progress()
    assert progress["searches"][0]["outstanding_count"] is None
    assert not progress["full_text_documents"]


def test_passage_windows_must_cover_every_character_for_full_text():
    from papyrus_chat.artifact.records import SourceReference
    from papyrus_chat.corpus.passages import DocumentPassagePage, PassageWindow

    ledger = EvidenceLedger()

    def page(start, end, call_id):
        result = DocumentPassagePage(
            document_id="doc",
            snapshot_id="snapshot",
            passage_count=1,
            windows=(
                PassageWindow(
                    passage_id="p",
                    passage_index=0,
                    text="α" * (end - start),
                    char_start=start,
                    char_end=end,
                    passage_length=10,
                    kind="edition",
                    source=SourceReference(repository_url="repo", commit="commit", path="doc"),
                ),
            ),
        )
        ledger.ingest(exchange("read_document_passages", {"document_id": "doc"}, result, call_id))

    page(0, 4, "a")
    page(6, 10, "b")
    assert not ledger.progress()["full_text_documents"]
    page(2, 8, "c")
    assert ledger.progress()["full_text_documents"] == ["doc"]


def test_memory_pages_and_notes_do_not_become_new_evidence():
    import asyncio
    from types import SimpleNamespace
    from typing import Any

    from pydantic_ai import RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    from papyrus_chat.agent.context.memory import (
        ResearchNotes,
        read_research_record,
        update_research_notes,
    )

    ledger = EvidenceLedger()
    ledger.ingest(exchange("search_documents", {"query": {}}, search_result(["doc-1"]), "original"))
    deps = SimpleNamespace(research_state=SimpleNamespace(ledger=ledger))
    ctx: RunContext[Any] = RunContext(deps=deps, model=TestModel(), usage=RunUsage())
    result = asyncio.run(read_research_record(ctx, ledger.records[0].record_id))
    assert result["fragments"] and result["next_offset"] is None
    notes = ResearchNotes(notes="Invented citation https://papyri.info/ddbdp/invented")
    asyncio.run(update_research_notes(ctx, notes))
    ledger.ingest(exchange("read_research_record", {}, result, "recall"))
    ledger.ingest(exchange("update_research_notes", {}, notes, "notes"))
    assert len(ledger.records) == 1
    assert ledger.notes == notes.notes
    assert "https://papyri.info/ddbdp/invented" not in ledger.corpus_urls


def test_coverage_note_distinguishes_candidates_excerpts_and_unknown_pages():
    from papyrus_chat.agent.context.coverage import coverage_note

    ledger = EvidenceLedger()
    assert coverage_note(ledger) == ""
    result = search_result(["doc-1"], next_offset=1)
    ledger.ingest(exchange("search_documents", {"query": {}}, result, "first"))
    ledger.ingest(exchange("search_documents", {"query": {}}, result, "repeat"))
    note = coverage_note(ledger)
    assert "1 unique candidate documents" in note
    assert "0 documents with inspected excerpts" in note
    assert "0 documents with all stored passage text" in note
    assert "outstanding in 1 searches" in note
    assert "incomplete" not in note
    legacy = search_result(["doc-2"])
    legacy.pop("offset")
    legacy.pop("next_offset")
    ledger.ingest(exchange("search_documents", {"query": {}}, legacy, "legacy"))
    assert "unknown for 1 searches" in coverage_note(ledger)
