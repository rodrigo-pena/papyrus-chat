"""MCP server contract tests."""

import asyncio
import os
import sys
from pathlib import Path

from tests.retrieval.test_discovery import content_service as content_service

from mcp import Client, StdioServerParameters
from papyrus_chat.corpus import CorpusService
from papyrus_chat.mcp_server import app as mcp_app
from papyrus_chat.mcp_server import create_mcp_server
from papyrus_chat.retrieval.discovery.models import DiscoveryQuery


def _tool_names(tools: list) -> list[str]:
    return [tool.name for tool in tools]


EXPECTED_TOOLS = [
    "get_corpus_info",
    "suggest_subjects",
    "search_documents",
    "facet_documents",
    "lookup_document",
    "discover_documents",
    "inspect_documents",
]


def test_mcp_server_lists_exactly_seven_annotated_tools(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    async def exercise() -> None:
        async with Client(create_mcp_server(service)) as client:
            result = await client.list_tools()
            assert _tool_names(result.tools) == EXPECTED_TOOLS
            for tool in result.tools:
                assert tool.annotations is not None
                assert tool.annotations.read_only_hint is True
                assert tool.annotations.destructive_hint is False
                assert tool.annotations.idempotent_hint is True
                assert tool.annotations.open_world_hint is False
                assert tool.output_schema is not None

    try:
        asyncio.run(exercise())
    finally:
        service.close()


def test_mcp_server_returns_structured_results_for_every_tool(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    async def exercise() -> None:
        async with Client(create_mcp_server(service)) as client:
            info = await client.call_tool("get_corpus_info", {})
            assert info.is_error is False
            assert info.structured_content["artifact_schema_version"] == 4

            suggestions = await client.call_tool(
                "suggest_subjects", {"concept": "taxes", "scope": {}, "limit": 3}
            )
            assert suggestions.is_error is False
            assert suggestions.structured_content["available"] is False

            search = await client.call_tool(
                "search_documents",
                {"query": {"term_groups": [["not-present"]], "limit": 2}},
            )
            assert search.is_error is False
            assert search.structured_content["candidate_count"] == 0

            facets = await client.call_tool(
                "facet_documents", {"query": {}, "field": "collection", "limit": 1}
            )
            assert facets.is_error is False
            assert facets.structured_content["truncated"] is True

            lookup = await client.call_tool(
                "lookup_document", {"identifier": "TM 999999", "limit": 1}
            )
            assert lookup.is_error is False
            assert lookup.structured_content["exact_match_count"] == 0

            inspection = await client.call_tool(
                "inspect_documents",
                {
                    "document_ids": ["missing:document"],
                    "excerpt_limit": 1,
                    "excerpt_chars": 200,
                },
            )
            assert inspection.is_error is False
            assert inspection.structured_content["missing"] == ["missing:document"]

            discovery = await client.call_tool(
                "discover_documents", {"query": {"text": "judicial complaints"}}
            )
            assert discovery.is_error is False
            assert discovery.structured_content["available"] is False
            assert discovery.structured_content["unavailable_reason"]
            assert "candidate_count" not in discovery.structured_content

    try:
        asyncio.run(exercise())
    finally:
        service.close()


def test_mcp_discovery_matches_service_and_chunk_inspection_works(content_service) -> None:
    async def exercise() -> None:
        async with Client(create_mcp_server(content_service)) as client:
            result = await client.call_tool(
                "discover_documents",
                {"query": {"text": "judicial complaints", "limit": 2}},
            )
            assert result.is_error is False
            content = result.structured_content
            assert content["available"] is True
            assert content["scope_document_count"] == 5
            assert content["indexed_document_count"] == 5
            assert len(content["hits"]) == 2
            assert "candidate_count" not in content

            direct = content_service.discover_documents(
                DiscoveryQuery(text="judicial complaints", limit=2)
            )
            assert content["hits"][0]["document_id"] == direct.hits[0].document_id
            assert content["hits"][0]["channels"] == list(direct.hits[0].channels)
            assert content["hits"][0]["chunks"][0]["chunk_id"] == (
                direct.hits[0].chunks[0].chunk_id
            )
            assert content["hits"][0]["canonical_url"] == direct.hits[0].canonical_url

            chunk_ids = [chunk["chunk_id"] for chunk in content["hits"][0]["chunks"]]
            inspection = await client.call_tool(
                "inspect_documents",
                {
                    "document_ids": [direct.hits[0].document_id],
                    "chunk_ids": chunk_ids,
                    "excerpt_limit": 1,
                },
            )
            assert inspection.is_error is False
            structured = inspection.structured_content
            assert structured["missing"] == []
            excerpt = structured["inspections"][0]["passages"][0]
            assert excerpt["chunk_id"] == chunk_ids[0]
            assert excerpt["source"]["path"] == direct.hits[0].chunks[0].source.path

    asyncio.run(exercise())


def test_mcp_server_converts_invalid_input_to_tool_error(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    async def exercise() -> None:
        async with Client(create_mcp_server(service)) as client:
            result = await client.call_tool("lookup_document", {"identifier": "", "limit": 20})
            assert result.is_error is True
            assert result.structured_content is None

    try:
        asyncio.run(exercise())
    finally:
        service.close()


def test_mcp_cli_help_does_not_require_the_optional_import() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(mcp_app, ["--help"])

    assert result.exit_code == 0
    assert "artifact" in result.output.lower()
    assert "stdio" in result.output.lower()


def test_real_stdio_process_initializes_and_serves_all_tools(corpus_artifact: Path) -> None:
    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "papyrus_chat.mcp_server", "--artifact", str(corpus_artifact)],
            cwd=Path.cwd(),
            env={key: value for key, value in os.environ.items() if not key.startswith("LLM_")},
        )
        async with Client(parameters) as client:
            tools = await client.list_tools()
            assert _tool_names(tools.tools) == EXPECTED_TOOLS
            assert (await client.call_tool("get_corpus_info", {})).is_error is False
            assert (
                await client.call_tool("suggest_subjects", {"concept": "taxes", "scope": {}})
            ).is_error is False
            assert (
                await client.call_tool("search_documents", {"query": {"term_groups": [["tax"]]}})
            ).is_error is False
            assert (
                await client.call_tool("facet_documents", {"query": {}, "field": "collection"})
            ).is_error is False
            assert (
                await client.call_tool("lookup_document", {"identifier": "TM 23944"})
            ).is_error is False
            assert (
                await client.call_tool("inspect_documents", {"document_ids": ["missing:document"]})
            ).is_error is False
            assert (
                await client.call_tool("discover_documents", {"query": {"text": "complaints"}})
            ).is_error is False

    asyncio.run(exercise())
