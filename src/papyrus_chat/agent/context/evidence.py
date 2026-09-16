"""Exact tool evidence, separate from lossy model-written summaries."""

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    NativeToolCallPart,
    NativeToolReturnPart,
    ToolCallPart,
    ToolReturnPart,
)

from papyrus_chat.corpus.models import (
    CorpusDescription,
    CorpusFacetResult,
    CorpusInspectionOutcome,
    CorpusSearchSummary,
    CorpusSubjectSuggestionSummary,
)
from papyrus_chat.corpus.passages import DocumentPassagePage
from papyrus_chat.retrieval.discovery.models import DiscoveryResult

from .progress import research_progress


def result_schemas() -> dict[str, type[BaseModel]]:
    # Loaded after agent dependencies, avoiding a tools -> state -> web -> tools cycle.
    from papyrus_chat.agent.web import WebBackgroundResult

    return {
        "describe_corpus": CorpusDescription,
        "search_documents": CorpusSearchSummary,
        "discover_documents": DiscoveryResult,
        "inspect_documents": CorpusInspectionOutcome,
        "read_document_passages": DocumentPassagePage,
        "facet_documents": CorpusFacetResult,
        "suggest_subject_values": CorpusSubjectSuggestionSummary,
        "search_web_background": WebBackgroundResult,
    }


@dataclass(frozen=True)
class EvidenceRecord:
    tool_name: str
    call_id: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @cached_property
    def record_id(self) -> str:
        payload = json.dumps(
            [self.tool_name, self.arguments, self.result], ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def render(self) -> str:
        return json.dumps(
            {
                "record_id": self.record_id,
                "tool": self.tool_name,
                "call_id": self.call_id,
                "source": "web background"
                if self.tool_name in {"search_web_background", "web_search"}
                else "corpus",
                "arguments": self.arguments,
                "result": self.result,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass
class EvidenceLedger:
    records: list[EvidenceRecord] = field(default_factory=list)
    corpus_urls: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)
    executions: list[tuple[str, str]] = field(default_factory=list)
    notes: str = ""
    record_ids: set[str] = field(default_factory=set)

    def progress(self) -> dict[str, Any]:
        return research_progress(self.records)

    def ingest(self, messages: list[ModelMessage]) -> None:
        calls: dict[str, ToolCallPart | NativeToolCallPart] = {}
        schemas = result_schemas()
        for message in messages:
            for part in message.parts:
                if isinstance(part, (ToolCallPart, NativeToolCallPart)):
                    calls[part.tool_call_id] = part
                    continue
                if not isinstance(part, (ToolReturnPart, NativeToolReturnPart)):
                    continue
                call = calls.pop(part.tool_call_id, None)
                if call is None or call.tool_name != part.tool_name:
                    continue
                try:
                    if part.tool_name == "update_research_notes":
                        from .memory import ResearchNotes

                        parsed_notes = (
                            ResearchNotes.model_validate_json(part.content)
                            if isinstance(part.content, str)
                            else ResearchNotes.model_validate(part.content)
                        )
                        notes_key = f"notes:{part.tool_call_id}:{parsed_notes.notes}"
                        if notes_key not in self.seen:
                            self.notes = parsed_notes.notes
                            self.seen.add(notes_key)
                        continue
                    if isinstance(part, NativeToolReturnPart):
                        if (
                            part.tool_name != "web_search"
                            or not isinstance(call, NativeToolCallPart)
                            or not isinstance(part.content, dict)
                        ):
                            continue
                        result = part.content
                    else:
                        schema = schemas.get(part.tool_name)
                        if schema is None or not isinstance(call, ToolCallPart):
                            continue
                        parsed = (
                            schema.model_validate_json(part.content)
                            if isinstance(part.content, str)
                            else schema.model_validate(part.content)
                        )
                        result = parsed.model_dump(mode="json")
                    record = EvidenceRecord(
                        part.tool_name,
                        part.tool_call_id,
                        call.args_as_dict(),
                        result,
                    )
                    key = f"{record.call_id}:{record.record_id}"
                except (ValidationError, ValueError, TypeError):
                    continue
                if key in self.seen:
                    continue
                self.seen.add(key)
                self.executions.append((record.call_id, record.record_id))
                if record.record_id not in self.record_ids:
                    self.records.append(record)
                    self.record_ids.add(record.record_id)
                if part.tool_name == "read_document_passages" and record.result.get(
                    "canonical_url"
                ):
                    self.corpus_urls.add(record.result["canonical_url"])
                if part.tool_name in {
                    "search_documents",
                    "discover_documents",
                    "inspect_documents",
                }:
                    for item in record.result.get("hits", record.result.get("inspections", [])):
                        if url := item.get("canonical_url"):
                            self.corpus_urls.add(url)
