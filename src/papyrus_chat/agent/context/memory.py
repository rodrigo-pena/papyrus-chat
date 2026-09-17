"""Request-local memory tools. Recalled data never becomes new source evidence."""

import json
from collections.abc import Iterator
from typing import Annotated, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from papyrus_chat.agent.tools import CorpusToolDeps

from .progress import progress_overview

ResearchNotesText = Annotated[
    str,
    Field(
        max_length=4000,
        description="Objective, interpretations, uncertainties, and remaining work; not evidence.",
    ),
]


class ResearchNotes(BaseModel):
    notes: ResearchNotesText


def record_fragments(value: Any, path: tuple[str | int, ...] = ()) -> Iterator[dict[str, Any]]:
    """Keep small facts whole; expose large text as exact, explicitly located windows."""
    if len(json.dumps(value, ensure_ascii=False).encode()) <= 4000:
        yield {"path": path, "value": value}
    elif isinstance(value, str):
        for start in range(0, len(value), 1500):
            yield {
                "path": path,
                "text": value[start : start + 1500],
                "char_start": start,
                "char_end": min(len(value), start + 1500),
                "total_chars": len(value),
            }
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from record_fragments(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from record_fragments(item, (*path, index))
    else:
        yield {"path": path, "value": value}


async def list_research_records(
    ctx: RunContext["CorpusToolDeps"],
    offset: Annotated[int, Field(ge=0)] = 0,
    document_id: str | None = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """List original evidence records; use their IDs to recall earlier exact evidence."""
    records = ctx.deps.research_state.ledger.records
    if tool_name:
        records = [r for r in records if r.tool_name == tool_name]
    if document_id:
        records = [r for r in records if document_id in r.render()]
    selected = records[offset : offset + 20]
    return {
        "total": len(records),
        "next_offset": offset + len(selected) if offset + len(selected) < len(records) else None,
        "records": [
            {
                "record_id": r.record_id,
                "tool": r.tool_name,
                "source": "web background" if "web" in r.tool_name else "corpus",
            }
            for r in selected
        ],
    }


async def read_research_record(
    ctx: RunContext["CorpusToolDeps"],
    record_id: str,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """Recall an original record in pages of exact fields or located text windows.

    Paths identify fields in the original record. Continue next_offset until absent.
    This is recalled evidence, not a new search or newly inspected corpus text.
    """
    record = next(
        (r for r in ctx.deps.research_state.ledger.records if r.record_id == record_id), None
    )
    if record is None:
        raise ModelRetry("Unknown record ID. Use list_research_records for this conversation.")
    fragments = list(record_fragments(json.loads(record.render())))
    selected = fragments[offset : offset + 5]
    return {
        "record_id": record_id,
        "tool": record.tool_name,
        "fragments": selected,
        "total_fragments": len(fragments),
        "next_offset": offset + len(selected) if offset + len(selected) < len(fragments) else None,
    }


async def get_research_progress(
    ctx: RunContext["CorpusToolDeps"], offset: Annotated[int, Field(ge=0)] = 0
) -> dict[str, Any]:
    """Get measured search and text coverage plus saved notes. Page through searches."""
    ledger = ctx.deps.research_state.ledger
    progress = ledger.progress()
    searches = progress["searches"][offset : offset + 10]
    return {
        **progress_overview(progress),
        "notes": ledger.notes,
        "search_details": searches,
        "next_offset": offset + len(searches)
        if offset + len(searches) < len(progress["searches"])
        else None,
    }


async def update_research_notes(
    ctx: RunContext["CorpusToolDeps"], notes: ResearchNotesText
) -> ResearchNotes:
    """Replace research notes before long investigations; preserve outstanding work.

    Use at most 4,000 characters. If too long, shorten and retry with the same
    {"notes": "..."} shape; rejected updates leave saved notes unchanged.
    Prioritize the objective, conclusions, completed/rejected directions, and
    remaining work. Recall detailed quotations from research records instead.
    Notes are model-written interpretations and never establish citation eligibility.
    """
    result = ResearchNotes(notes=notes)
    ctx.deps.research_state.ledger.notes = result.notes
    return result


def register_memory_tools(agent: Agent[Any, Any]) -> None:
    agent.tool(list_research_records)
    agent.tool(read_research_record)
    agent.tool(get_research_progress)
    agent.tool(update_research_notes)
