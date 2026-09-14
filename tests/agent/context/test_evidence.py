from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from papyrus_chat.agent.context.evidence import EvidenceLedger

URL = "https://papyri.info/ddbdp/p.mich;8;480"


def history(content=None):
    result = (
        content
        if content is not None
        else {
            "query": {},
            "candidate_count": 7,
            "truncated": False,
            "hits": [
                {
                    "document_id": "doc-1",
                    "title": "Document",
                    "collection": "ddbdp",
                    "languages": ["grc"],
                    "canonical_url": URL,
                }
            ],
        }
    )
    return [
        ModelResponse(parts=[ToolCallPart("search_documents", {"query": {}}, "search-1")]),
        ModelRequest(parts=[ToolReturnPart("search_documents", result, "search-1")]),
    ]


def test_evidence_retains_exact_results_and_scope():
    ledger = EvidenceLedger()
    ledger.ingest(history())
    ledger.ingest(history())
    assert ledger.corpus_urls == {URL}
    assert len(ledger.records) == 1
    assert ledger.records[0].result["candidate_count"] == 7
    assert ledger.records[0].arguments == {"query": {}}


def test_summary_prose_and_invalid_or_orphaned_results_are_not_citation_evidence():
    ledger = EvidenceLedger()
    ledger.ingest([ModelResponse(parts=[TextPart(URL)])])
    ledger.ingest(history({"hits": [{"canonical_url": URL}]}))
    ledger.ingest(history()[1:])
    assert not ledger.corpus_urls
    assert not ledger.records


def test_ledgers_do_not_share_evidence():
    first, second = EvidenceLedger(), EvidenceLedger()
    first.ingest(history())
    assert not second.records
    assert not second.corpus_urls


def test_native_web_sources_are_preserved_as_background_not_corpus_evidence():
    from pydantic_ai.messages import NativeToolCallPart, NativeToolReturnPart

    ledger = EvidenceLedger()
    ledger.ingest(
        [
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        "web_search", {"query": "reign dates"}, "web-1", provider_name="openai"
                    ),
                    NativeToolReturnPart(
                        "web_search",
                        {"status": "completed", "sources": [{"url": URL}]},
                        "web-1",
                        provider_name="openai",
                    ),
                ]
            )
        ]
    )
    assert len(ledger.records) == 1
    assert "web background" in ledger.records[0].render()
    assert URL in ledger.records[0].render()
    assert not ledger.corpus_urls
