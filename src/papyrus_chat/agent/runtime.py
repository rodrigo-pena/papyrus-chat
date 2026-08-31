"""Pydantic AI runtime for the evidence-grounded research agent."""

import re
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext, WebSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService, register_corpus_tools
from papyrus_chat.chat.provider import ProviderConfig

RESEARCH_INSTRUCTIONS = """
You are an evidence-grounded papyrologist research assistant.

Use the read-only corpus tools for every claim about a corpus document or
transcription. First infer and disclose the Scope and method: the collections,
inclusive date interval, transcription language, and multilingual lexical term
groups you searched. Treat candidate counts as exact for the displayed filters,
not as an exhaustive scholarly classification. Cite each corpus document with
the papyri.info URL returned by a corpus tool and distinguish transcription
evidence from model-generated synthesis. Label calendar, historical, or other
knowledge not present in corpus results as model-supplied background. If native
web search is available, preserve its provider citations and never use web
results as a replacement for local corpus evidence.
""".strip()

_PAPYRI_URL = re.compile(r"https://papyri\.info/[^\s)\]>]+")
_NO_CORPUS_EVIDENCE = re.compile(
    r"\b(?:no|none|not any|without|insufficient)\b.{0,40}\bcorpus evidence\b",
    re.IGNORECASE,
)


def model_supports_native_web_search(model_name: str) -> bool:
    """OpenAI native web search is available through the Responses model prefix."""
    return model_name.startswith("openai-responses:")


def validate_research_output(output: str, known_corpus_urls: set[str]) -> str:
    """Require corpus links in answers that claim to present corpus evidence."""
    urls = {url.rstrip(".,") for url in _PAPYRI_URL.findall(output)}
    unknown = sorted(url for url in urls if url not in known_corpus_urls)
    if unknown:
        raise ModelRetry(
            "Use only a known corpus citation returned by a corpus tool; "
            f"unknown corpus citation: {unknown[0]}"
        )
    if (
        "corpus evidence" in output.casefold()
        and not urls
        and not _NO_CORPUS_EVIDENCE.search(output)
    ):
        raise ModelRetry(
            "Corpus evidence must include a known corpus citation returned by a corpus tool."
        )
    return output


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

    if capabilities:
        agent = Agent[CorpusToolDeps, str](
            selected_model,
            deps_type=CorpusToolDeps,
            output_type=str,
            instructions=RESEARCH_INSTRUCTIONS,
            capabilities=capabilities,
        )
    else:
        agent = Agent[CorpusToolDeps, str](
            selected_model,
            deps_type=CorpusToolDeps,
            output_type=str,
            instructions=RESEARCH_INSTRUCTIONS,
        )
    register_corpus_tools(agent)

    @agent.output_validator
    def validate_output(ctx: RunContext[CorpusToolDeps], output: str) -> str:
        return validate_research_output(output, ctx.deps.known_corpus_urls)

    return agent
