"""Exact tool evidence, separate from lossy model-written summaries."""

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
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
from papyrus_chat.retrieval.discovery.models import DiscoveryResult


def result_schemas() -> dict[str, type[BaseModel]]:
    # Loaded after agent dependencies, avoiding a tools -> state -> web -> tools cycle.
    from papyrus_chat.agent.web import WebBackgroundResult

    return {
        "describe_corpus": CorpusDescription,
        "search_documents": CorpusSearchSummary,
        "discover_documents": DiscoveryResult,
        "inspect_documents": CorpusInspectionOutcome,
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

    def render(self) -> str:
        return json.dumps(
            {
                "tool": self.tool_name,
                "call_id": self.call_id,
                "source": "web background"
                if self.tool_name == "search_web_background"
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

    def ingest(self, messages: list[ModelMessage]) -> None:
        calls: dict[str, ToolCallPart] = {}
        schemas = result_schemas()
        for message in messages:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        calls[part.tool_call_id] = part
            elif isinstance(message, ModelRequest):
                for part in message.parts:
                    if not isinstance(part, ToolReturnPart):
                        continue
                    call = calls.pop(part.tool_call_id, None)
                    schema = schemas.get(part.tool_name)
                    if call is None or call.tool_name != part.tool_name or schema is None:
                        continue
                    try:
                        content = part.content
                        parsed = (
                            schema.model_validate_json(content)
                            if isinstance(content, str)
                            else schema.model_validate(content)
                        )
                        record = EvidenceRecord(
                            part.tool_name,
                            part.tool_call_id,
                            call.args_as_dict(),
                            parsed.model_dump(mode="json"),
                        )
                    except (ValidationError, ValueError, TypeError):
                        continue
                    key = record.render()
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    self.records.append(record)
                    if part.tool_name in {
                        "search_documents",
                        "discover_documents",
                        "inspect_documents",
                    }:
                        for item in record.result.get("hits", record.result.get("inspections", [])):
                            if url := item.get("canonical_url"):
                                self.corpus_urls.add(url)
