"""Coverage computed from original returned records, never from model assertions."""

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evidence import EvidenceRecord


def merged_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def covered_length(ranges: list[tuple[int, int]]) -> int:
    return sum(right - left for left, right in merged_ranges(ranges))


def research_progress(records: list["EvidenceRecord"]) -> dict[str, Any]:
    candidates: set[str] = set()
    excerpts: set[str] = set()
    searches: dict[str, dict[str, Any]] = {}
    passages: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        result = record.result
        if record.tool_name in {"search_documents", "discover_documents"}:
            query = {k: v for k, v in result["query"].items() if k not in {"offset", "limit"}}
            key = record.tool_name + json.dumps(query, sort_keys=True, ensure_ascii=False)
            search = searches.setdefault(
                key,
                {
                    "tool": record.tool_name,
                    "query": query,
                    "total": None,
                    "ranges": [],
                    "documents": set(),
                    "legacy": False,
                },
            )
            ids = {hit["document_id"] for hit in result.get("hits", [])}
            candidates.update(ids)
            search["documents"].update(ids)
            count = result.get("candidate_count", result.get("ranked_candidate_count"))
            offset = result.get("offset")
            if offset is None or count is None:
                search["legacy"] = True
            else:
                # Mixed totals cannot establish complete coverage of one snapshot.
                if search["total"] is not None and search["total"] != count:
                    search["legacy"] = True
                search["total"] = count
                search["ranges"].append((offset, offset + len(result.get("hits", []))))
            if record.tool_name == "discover_documents":
                search["scope_document_count"] = result.get("scope_document_count")
                search["indexed_document_count"] = result.get("indexed_document_count")
        elif record.tool_name == "inspect_documents":
            excerpts.update(i["document_id"] for i in result["inspections"] if i["passages"])
        elif record.tool_name == "read_document_passages":
            doc = result["document_id"]
            entry = passages.setdefault(
                (result["snapshot_id"], doc), {"count": result["passage_count"], "passages": {}}
            )
            for window in result["windows"]:
                excerpts.add(doc)
                passage = entry["passages"].setdefault(
                    window["passage_id"], {"length": window["passage_length"], "ranges": []}
                )
                passage["ranges"].append((window["char_start"], window["char_end"]))
    full = sorted(
        {
            doc
            for (_, doc), entry in passages.items()
            if entry["count"] > 0
            and len(entry["passages"]) == entry["count"]
            and all(covered_length(p["ranges"]) == p["length"] for p in entry["passages"].values())
        }
    )
    summaries = []
    for search in searches.values():
        total = search.pop("total")
        ranges = search.pop("ranges")
        legacy = search.pop("legacy")
        ids = sorted(search.pop("documents"))
        retrieved = min(len(ids), covered_length(ranges))
        summaries.append(
            {
                **search,
                "candidate_count": total,
                "retrieved_count": len(ids),
                "page_ranges": merged_ranges(ranges),
                "outstanding_count": None if legacy or total is None else max(0, total - retrieved),
            }
        )
    return {
        "candidate_documents": sorted(candidates),
        "excerpt_documents": sorted(excerpts),
        "full_text_documents": full,
        "searches": summaries,
    }


def progress_overview(progress: dict[str, Any]) -> dict[str, Any]:
    searches = progress["searches"]
    return {
        "candidate_documents": len(progress["candidate_documents"]),
        "excerpt_documents": len(progress["excerpt_documents"]),
        "full_text_documents": len(progress["full_text_documents"]),
        "searches": len(searches),
        "searches_with_unread_results": sum(bool(s["outstanding_count"]) for s in searches),
        "searches_with_unknown_coverage": sum(s["outstanding_count"] is None for s in searches),
    }
