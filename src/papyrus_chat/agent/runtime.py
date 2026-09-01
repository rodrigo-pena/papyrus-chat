"""Pydantic AI runtime for the evidence-grounded research agent."""

import re
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext, WebSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService, register_corpus_tools
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.retrieval.structured import CorpusDocumentMatch

RESEARCH_INSTRUCTIONS = """
You are an evidence-grounded papyrologist research assistant.

Use the read-only corpus tools for every claim about a corpus document or
transcription. The search tool returns lean summaries: exact candidate counts,
located snippets with line references, and citation URLs. Call inspect_documents
on selected document ids for bounded excerpts and HGV context, and facet_documents
to size a refinement before searching. First infer and disclose the Scope and
method: the collections, inclusive date interval, transcription language, and
multilingual lexical term groups you searched. Treat candidate counts as exact
for the displayed filters, not as an exhaustive scholarly classification. Cite
each corpus document with the papyri.info URL exactly as a corpus tool returned
it: never build a citation from a document title, identifier, or memory, and
treat a document as citable only once search_documents or inspect_documents has
returned it in this conversation. Distinguish transcription evidence from
model-generated synthesis. Label calendar, historical, or other knowledge not
present in corpus results as model-supplied background. If native web search is
available, preserve its provider citations and never use web results as a
replacement for local corpus evidence.
""".strip()

_PAPYRI_URL = re.compile(r"https://papyri\.info/[^\s)\]>]+")
# Trailing prose punctuation and invisible characters a model may attach to a
# copied URL; a real papyri.info citation always ends in an alphanumeric.
_URL_TRIM = ".,;:!?\"'`’”«»}]…|" + "\u200b\u200c\u200d\u2060\ufeff"
_NO_CORPUS_EVIDENCE = re.compile(
    r"\b(?:no|none|not any|without|insufficient)\b.{0,40}\bcorpus evidence\b",
    re.IGNORECASE,
)


def model_supports_native_web_search(model_name: str) -> bool:
    """OpenAI native web search is available through the Responses model prefix."""
    return model_name.startswith("openai-responses:")


def validate_research_output(
    output: str,
    known_corpus_urls: set[str],
    *,
    citation_lookup: Callable[[str], CorpusDocumentMatch | None] | None = None,
) -> str:
    """Require corpus links in answers that claim to present corpus evidence."""
    urls = {url.rstrip(_URL_TRIM) for url in _PAPYRI_URL.findall(output)}
    unknown = sorted(url for url in urls if url not in known_corpus_urls)
    if unknown:
        raise ModelRetry(_unknown_citations_message(unknown, known_corpus_urls, citation_lookup))
    if (
        "corpus evidence" in output.casefold()
        and not urls
        and not _NO_CORPUS_EVIDENCE.search(output)
    ):
        raise ModelRetry(
            "Corpus evidence must include a known corpus citation returned by a corpus tool."
        )
    return output


def _unknown_citations_message(
    unknown: list[str],
    known_corpus_urls: set[str],
    citation_lookup: Callable[[str], CorpusDocumentMatch | None] | None,
) -> str:
    listed = ", ".join(unknown[:3])
    if len(unknown) > 3:
        listed += f", and {len(unknown) - 3} more"
    return (
        "Use only known corpus citations returned by a corpus tool. "
        f"Unknown citation(s): {listed}. "
        + _unknown_citation_guidance(unknown[0], known_corpus_urls, citation_lookup)
    )


def _unknown_citation_guidance(
    citation: str,
    known_corpus_urls: set[str],
    citation_lookup: Callable[[str], CorpusDocumentMatch | None] | None,
) -> str:
    """Explain why a citation was rejected and what to cite instead."""
    if citation_lookup is None:
        return (
            "Cite only papyri.info URLs returned by search_documents or "
            "inspect_documents in this conversation."
        )
    document = citation_lookup(citation)
    if document is not None:
        return (
            f"{citation} is corpus document {document.document_id} ({document.title}), "
            "but no corpus tool returned it in this conversation. Search or inspect "
            "that document first, then cite the URL exactly as the tool returns it."
        )
    guidance = (
        f"{citation} is not a corpus document URL. Cite papyri.info URLs exactly as "
        "returned by search_documents or inspect_documents; never construct one from "
        "a document title, identifier, or memory."
    )
    suggestions = _series_suggestions(citation, known_corpus_urls)
    if suggestions:
        guidance += f" Known citations for this series: {', '.join(suggestions)}."
    return guidance


def _series_suggestions(citation: str, known_corpus_urls: set[str], limit: int = 3) -> list[str]:
    """Return known citations sharing the cited URL's publication series."""
    series = citation.rsplit("/", 1)[-1].split(";", 1)[0]
    if not series:
        return []
    marker = f"/{series};"
    return sorted(url for url in known_corpus_urls if marker in url)[:limit]


def create_research_agent(
    config: ProviderConfig,
    service: CorpusToolService,
    *,
    model: Any | None = None,
    enable_native_web_search: bool = True,
) -> Agent[Any, str]:
    """Construct an agent using the existing provider environment contract."""
    capabilities: list[NativeTool] = []
    selected_model = model
    if selected_model is None:
        api_key = config.api_key.get_secret_value() if config.api_key is not None else None
        provider = OpenAIProvider(base_url=config.base_url, api_key=api_key)
        if enable_native_web_search and model_supports_native_web_search(config.model):
            selected_model = OpenAIResponsesModel(
                config.model.removeprefix("openai-responses:"), provider=provider
            )
            capabilities.append(NativeTool(WebSearchTool()))
        else:
            selected_model = OpenAIChatModel(config.model, provider=provider)

    agent = Agent[CorpusToolDeps, str](
        selected_model,
        deps_type=CorpusToolDeps,
        output_type=str,
        instructions=RESEARCH_INSTRUCTIONS,
        capabilities=capabilities or None,
        retries=3,
    )
    register_corpus_tools(agent)

    @agent.output_validator
    def validate_output(ctx: RunContext[CorpusToolDeps], output: str) -> str:
        return validate_research_output(
            output,
            ctx.deps.known_corpus_urls,
            citation_lookup=ctx.deps.service.document_for_citation,
        )

    return agent
