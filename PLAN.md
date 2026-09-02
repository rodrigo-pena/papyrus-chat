# Papyrologist Research Chat v2

## Summary

Replace the bespoke Jinja/FastAPI interface and hand-written LLM client with a Pydantic AI v2 agent served through its stock web chat. The local app gains streaming, persistent browser conversations, tool visualization, and responsive UI without maintaining frontend code. [Pydantic AI Web Chat UI](https://pydantic.dev/docs/ai/guides/web/)

Add first-class `ddbdp` support while retaining `dclp` and `translations`. DDbDP transcriptions will be joined to HGV dates and descriptive metadata through their identifiers. [idp.data collection descriptions](https://github.com/papyri/idp.data)

Do not maintain an Egyptian-calendar vocabulary or other domain glossary. The agent generates Greek/German query variants and background facts from model knowledge, optionally using provider-native web search when supported. It must distinguish that background knowledge from evidence found in the local corpus.

## Public Contracts

- `papyrus-corpus-build` accepts `ddbdp`; selecting it implicitly fetches `DDbDP/` and `HGV_meta_EpiDoc/`. HGV remains linked metadata rather than a user-visible collection.
- Artifact schema v3 replaces v2; old artifacts receive an actionable rebuild message rather than an in-place migration. It can bundle a portable HGV subject vocabulary index.
- `papyrus-chat` keeps its existing artifact, host, port, and provider environment interface, but requires reliable function/tool calling.
- The old `/search`, `/documents/...`, and `/chat` pages are removed. The stock UI owns `/`, `/{thread_id}`, `/api/chat`, `/api/configure`, and `/api/health`.
- New immutable interfaces:
    - `CorpusQuery`: collections, OR-within/AND-between term groups, fields, transcription languages, inferred date interval, and bounded result limit.
    - `CorpusHit`: document metadata, matched passage and locator, provenance, and canonical papyri.info URL.
    - `CorpusQueryResult`: normalized query, disclosed assumptions, exact candidate count, truncation state, and ranked hits.
- Read-only agent tools: `describe_corpus`, `search_documents`, `inspect_documents`, `facet_documents`, and `suggest_subject_values`. No arbitrary SQL, filesystem, or code-execution tool is exposed.
- Register Pydantic AI’s provider-native web-search capability where the configured model supports it; otherwise offer an explicit DuckDuckGo terminology fallback. Web results never contribute corpus counts or citations. [Native tool support](https://pydantic.dev/docs/ai/guides/web/#native-tool-support)

Every answer must:

- show the inferred date range, language, collections, and generated lexical terms under **Scope and method**;
- describe counts as exact for those displayed filters but not an exhaustive scholarly classification;
- label uncited calendar or historical knowledge as model-supplied background;
- preserve provider citations when web search was used;
- link every cited corpus document directly to a papyri.info URL returned by a corpus tool;
- separate transcription evidence from model-generated synthesis.

## Conventional Commit Sequence

1. `docs: specify the Pydantic AI research chat`

    Record the product contract, artifact-v2 boundary, exploratory-answer semantics, model/web background-knowledge policy, removed UI routes, and collaborator acceptance journey.

2. `feat(builder): parse DDbDP and linked HGV components`

    Add DDbDP and HGV adapters plus component boundary models. Extract DDbDP edition languages and line-aware passages; extract HGV titles, subjects, commentary, material, origins, and date intervals. Join through HGV identifiers while preserving missing and one-to-many links. Add paired real-source fixtures and parser tests.

3. `feat(artifact): publish documentary corpus schema v2`

    Add source-attributed component, metadata, date, and language tables; document-level FTS for distinct-document querying; passage FTS for located evidence; deterministic hashing and validation of all linked data. Wire `ddbdp` into the CLI with implicit HGV acquisition and reject schema-v1 artifacts with rebuild instructions.

4. `feat(retrieval): add structured multilingual corpus queries`

    Implement `CorpusQuery`, grouped lexical matching, field restrictions, language filters, inclusive date-overlap filters, facets, exact distinct candidate counts, stable ranking, and bounded evidence lookup. Queries remain language-agnostic: Greek, German, and English terms come from the agent rather than a maintained vocabulary.

5. `feat(agent): expose read-only corpus research tools`

    Wrap retrieval in the five typed Pydantic AI tools, including scoped semantic subject suggestions. Return the complete normalized query with each result so follow-up turns can reuse and refine "this corpus." Enforce limits on groups, terms, text length, result count, inspected documents, and excerpts.

6. `feat(agent): adopt the Pydantic AI runtime`

    Add `pydantic-ai-slim[openai,web]>=2,<3`; construct `OpenAIChatModel` with the existing custom base URL and optional API key; register provider-native web search where supported; add research instructions and a minimal startup tool-capability probe. Use an output validator to require known papyri.info citations for evidence-based answers while allowing clearly labelled model-memory background and provider-supplied web citations. [OpenAI-compatible configuration](https://pydantic.dev/docs/ai/models/openai/)

7. `feat(ui)!: replace the custom web interface with Pydantic chat`

    Serve `Agent.to_web()` at the root using the selected CDN-and-cache asset delivery. Update the CLI, inject the artifact-backed dependencies, and remove FastAPI/Jinja templates, CSS, multipart handling, the legacy provider/conversation implementation, and old route tests. Preserve localhost binding and secret isolation.

8. `test(eval): cover the papyrologist research dialogue`

    Add deterministic `FunctionModel` tests for multi-step tool use, prior-query refinement, term generation, date disclosure, corpus citations, insufficient evidence, and separation of background knowledge from corpus evidence. Globally disable accidental real model calls in the default suite. [Pydantic AI testing guidance](https://pydantic.dev/docs/ai/guides/testing/)

9. `docs: document DDbDP research and validation`

    Update the README with the full-corpus build command, first-run CDN requirement, tool-calling requirement, meaning of candidate counts, optional native web search, and the collaborator walkthrough. Record full-corpus build/search measurements and the feedback outcome.

Each implementation commit includes its focused tests and must leave `pytest`, Ruff, formatting, and ty checks green.

## Test and Acceptance Plan

- Builder tests cover DDbDP passages, actual transcription languages, HGV subjects/dates, absent and multiple links, canonical URLs, provenance, and deterministic rebuilds.
- Retrieval tests cover model-supplied multilingual term groups, date overlap, candidate counts, facets, stable ordering, bounded queries, and injection-shaped input.
- Agent tests use `TestModel`/`FunctionModel` to verify exact tool arguments, history reuse, citation validation, memory-only background labels, and simulated web citations.
- UI integration tests inject local minimal HTML to avoid CDN access, exercise the stock streaming API, and confirm secrets never enter responses or tool output.
- Full-corpus validation covers roughly 68,000 DDbDP and 67,000 HGV source files at a pinned commit.
- Final acceptance is the collaborator’s four-query conversation:
    - the first answer discloses how it interpreted the historical period and what multilingual terms it searched;
    - the second reuses the prior candidate set and derives Egyptian month names from model knowledge or web search;
    - the third groups tax categories with linked papyrus examples;
    - the fourth distinguishes attested evidence from broader historical inference;
    - every named corpus record opens on papyri.info.

## Assumptions and Non-goals

- Historical periods and calendar terminology are inferred by the agent and disclosed, not encoded as application reference data.
- Native web search is optional; generic OpenAI-compatible endpoints may provide only function calling, in which case the agent uses model knowledge.
- Corpus evidence always comes from the pinned local artifact; web results may inform terminology but never replace papyrus evidence.
- Counts describe the displayed lexical/structured query and are not advertised as exhaustive scholarly classifications.
- The application remains local, single-user, read-only, and CDN-dependent on first UI use.
- Document/passage embeddings, GraphRAG, persistent relevance feedback, a custom frontend, authentication, hosted deployment, and publication-grade completeness remain out of scope. Schema-v3 bundles an optional local HGV-subject vocabulary index for exploratory cohort discovery.
