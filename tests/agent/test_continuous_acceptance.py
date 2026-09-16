"""End-to-end local regression for sustained, evidence-preserving research."""

import json
import shutil

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.context.compaction import SUMMARY_INSTRUCTIONS
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


def test_sustained_research_recalls_exact_evidence_and_completes_after_recovery(
    corpus_artifact, tmp_path
):
    root = tmp_path / "corpus"
    shutil.copytree(corpus_artifact, root)
    search = StructuredCorpusSearch(root / "corpus.sqlite")
    passage_id, document_id = search._connection.execute(
        "SELECT passage_id, document_id FROM passages LIMIT 1"
    ).fetchone()
    greek = "δραχμὰς δέκα\n" * 1100
    search._connection.execute(
        "UPDATE passages SET display_text=? WHERE passage_id=?", (greek, passage_id)
    )
    service = CorpusService(search)
    deps = CorpusToolDeps(service)
    requests = summaries = searches = passage_pages = 0
    phase = "search"
    next_offset = 0
    cursor = None
    pending = None
    record_id = None
    recall_offset = 0
    recalled = []
    restored = ""
    candidate_ids = set()
    candidate_count = None
    citation = None
    exhausted = False
    recalled_after_compaction = False

    def dialogue(messages, info):
        nonlocal requests, summaries, searches, passage_pages, phase, next_offset, cursor
        nonlocal pending, record_id, recall_offset, restored, candidate_count, citation
        nonlocal exhausted, recalled_after_compaction
        if SUMMARY_INSTRUCTIONS in (info.instructions or ""):
            summaries += 1
            return ModelResponse(
                [TextPart("Continue the planned inventory and exact text review.")]
            )
        requests += 1
        assert requests < 80, "script failed to terminate"
        assert info.function_tools, "research must finish naturally"
        if pending and phase != "padding":
            returns = [
                p
                for m in messages
                for p in m.parts
                if isinstance(p, ToolReturnPart) and p.tool_call_id == pending
            ]
            assert returns, f"latest tool result disappeared: {pending}"
            content = returns[-1].content
            result = content.model_dump(mode="json") if isinstance(content, BaseModel) else content
            if isinstance(result, str):
                result = json.loads(result)
            if phase == "search":
                searches += 1
                candidate_ids.update(hit["document_id"] for hit in result["hits"])
                candidate_count = result["candidate_count"]
                next_offset = result["next_offset"]
                if next_offset is None:
                    phase = "passages"
            elif phase == "passages":
                passage_pages += 1
                citation = result["canonical_url"]
                restored += "".join(
                    w["text"] for w in result["windows"] if w["passage_id"] == passage_id
                )
                cursor = result["next_cursor"]
                if cursor is None:
                    phase = "padding"
            elif phase == "list":
                record_id = result["records"][0]["record_id"]
                phase = "recall"
            elif phase == "recall":
                recalled.extend(result["fragments"])
                recall_offset = result["next_offset"]
                if recall_offset is None:
                    recalled_after_compaction = summaries > 3
                    phase = "answer"
            pending = None
        if phase == "padding" and requests >= 21 and summaries > 3:
            phase = "list"
        if phase == "answer":
            if not exhausted:
                exhausted = True
                return ModelResponse(
                    [TextPart("UNFINISHED_DRAFT")],
                    finish_reason="length",
                    usage=RequestUsage(input_tokens=100, output_tokens=200),
                )
            return ModelResponse(
                [TextPart(f"Corpus evidence: {citation}. Inspected text reads δραχμὰς δέκα.")],
                usage=RequestUsage(input_tokens=110, output_tokens=30),
            )
        if phase == "search":
            tool, args = "search_documents", {"query": {"limit": 1, "offset": next_offset}}
        elif phase == "passages":
            tool, args = "read_document_passages", {"document_id": document_id, "cursor": cursor}
        elif phase == "list":
            tool, args = "list_research_records", {"tool_name": "search_documents"}
        elif phase == "recall":
            tool, args = "read_research_record", {"record_id": record_id, "offset": recall_offset}
        else:
            tool, args = "describe_corpus", {}
        pending = f"step-{requests}"
        return ModelResponse(
            [
                TextPart("Tentative interpretation πάπυρος " * (1000 if phase == "padding" else 1)),
                ToolCallPart(tool, args, pending),
            ]
        )

    agent = create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(context_window=65536),
    )
    try:
        result = agent.run_sync(
            "Page through the corpus and read the selected Greek document.", deps=deps
        )
        assert requests > 16
        assert summaries > 3
        assert searches > 1
        assert passage_pages > 1
        assert restored == greek
        assert len(candidate_ids) == candidate_count
        assert recalled_after_compaction
        assert recalled[0]["value"]["result"]["candidate_count"] == candidate_count
        assert recalled[0]["value"]["arguments"]["query"]["offset"] == 0
        assert deps.research_state.ledger.records[0].result["candidate_count"] == candidate_count
        assert exhausted
        assert result.usage.requests == requests + summaries
        assert result.usage.output_tokens >= 230
        assert "UNFINISHED_DRAFT" not in result.output
        assert citation is not None
        assert citation in result.output
        assert f"Coverage: {candidate_count} unique candidate documents retrieved" in result.output
        assert "1 documents with all stored passage text retrieved" in result.output
        assert "Result pages outstanding in 0 searches" in result.output
    finally:
        service.close()
