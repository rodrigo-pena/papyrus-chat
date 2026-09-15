# Papyrus Chat

Build a searchable, provenance-preserving corpus from
[papyri/idp.data](https://github.com/papyri/idp.data), then investigate it in
a local Pydantic AI chat. The assistant discloses its search scope, separates
local transcription evidence from web-sourced and model-supplied background,
and links cited records to papyri.info.

> See the [local MCP integration guide](docs/local-mcp.md) for instructions on
> setting up a local MCP server for interacting with Papyri.info data from your
> own chatbot or coding agent interface.

## Quick start

This is the lowest-effort route for using Papyrus Chat: install
[uv](https://docs.astral.sh/uv/getting-started/installation/), make sure Git is
available, and run the commands below. You do not need to clone this repository
or create a Python environment; `uvx` downloads Papyrus Chat from GitHub and
runs it in an isolated, cached environment.

Run these commands from the directory where you want to keep the corpus data:

```bash
# Download the semantic model snapshot once
uvx --from huggingface-hub hf download intfloat/multilingual-e5-small \
  --revision 4a4cddf9cf6d77a61cc1c73f824ec2127773db85 \
  --local-dir ./models/multilingual-e5-small

# Build a corpus artifact with semantic subject suggestions
uvx --from 'papyrus-chat[semantic,web] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-corpus-build dclp ddbdp translations \
  --semantic-model-dir ./models/multilingual-e5-small \
  --output ./data/papyrus-corpus

# Configure an OpenAI-compatible model
export LLM_BASE_URL="https://provider.example/v1"
export LLM_MODEL="model-name"
export LLM_API_KEY="..."   # optional for local, unauthenticated endpoints

# Start the local chat and open it in your browser
uvx --from 'papyrus-chat[semantic,web] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-chat --artifact ./data/papyrus-corpus --web-search
```

The build output is persistent; it is not stored in uv's tool cache. With the
commands above, `./data/papyrus-corpus` is relative to the directory from which
you ran the builder and contains:

```text
data/papyrus-corpus/
├── manifest.json
├── corpus.sqlite
└── ATTRIBUTION.md
```

The build step technically needs to be run only once. Reuse the same artifact
for every later chat session. Run the builder again only when you want to sync
with the current [papyri/idp.data](https://github.com/papyri/idp.data) state;
because the destination already exists, use `--force` to replace it:

```bash
uvx --from 'papyrus-chat[semantic,web] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-corpus-build dclp ddbdp translations \
  --semantic-model-dir ./models/multilingual-e5-small \
  --output ./data/papyrus-corpus --force
```

You can also skip the build entirely if you obtained a compatible artifact
elsewhere: keep its directory intact and pass that directory to the chat
command, for example:

```bash
uvx --from 'papyrus-chat[semantic,web] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-chat --artifact /path/to/papyrus-corpus --web-search
```

Semantic subject suggestions and contextual web search are enabled by default
in this quick start. If you do not want one of them, remove the corresponding
model download/`--semantic-model-dir` or `--web-search` option and change the
`uvx --from` requirement in both commands: use
`papyrus-chat[web] @ git+https://github.com/rodrigo-pena/papyrus-chat.git` without
semantic embeddings, `papyrus-chat[semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git`
without web search, or the bare Git URL without either extra.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git
- An OpenAI-compatible endpoint with function/tool calling and streaming
  support. Set its base URL and model with `LLM_BASE_URL` and `LLM_MODEL`; an
  API key is optional for unauthenticated local endpoints.
- Internet access the first time the browser UI is opened, so Pydantic AI can
  fetch and cache its stock chat UI from the CDN.

## Supported upstream collections

The upstream [papyri/idp.data](https://github.com/papyri/idp.data) repository
contains data from several projects. Papyrus Chat currently supports exactly
three user-selectable collections:

| CLI name       | Upstream directory                                                             | Content added to the artifact                                            |
| -------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `dclp`         | [`DCLP/`](https://github.com/papyri/idp.data/tree/master/DCLP)                 | Literary-papyrus metadata plus edition and embedded translation passages |
| `ddbdp`        | [`DDbDP/`](https://github.com/papyri/idp.data/tree/master/DDbDP)               | Documentary transcriptions, identifiers, and metadata                    |
| `translations` | [`Translations/`](https://github.com/papyri/idp.data/tree/master/Translations) | Published translation passages and their record metadata                 |

Selecting `ddbdp` also reads [`HGV_meta_EpiDoc/`](https://github.com/papyri/idp.data/tree/master/HGV_meta_EpiDoc)
and links matching HGV dates and descriptive metadata to DDbDP documents. HGV
is an auxiliary source, not a fourth selectable collection. No other upstream
directory is currently ingested. The authoritative runtime list is `papyrus-corpus-build --list-collections`.

See the [collection adapter guide](docs/collection-adapters.md) for hints on how to add another upstream collection or an auxiliary linked source.

## From a project checkout

```bash
# 1. Install the locked environment
uv sync

# 2. Build the full selected documentary corpus (no LLM or credentials needed)
uv run papyrus-corpus-build dclp ddbdp translations --output ./data/papyrus-corpus

# 3. Configure the model and start the local chat
export LLM_BASE_URL="https://provider.example/v1"
export LLM_MODEL="model-name"
export LLM_API_KEY="..."   # optional if using local, unauthenticated servers
uv run papyrus-chat --artifact ./data/papyrus-corpus
```

Both commands write timestamped stage logs to the terminal. Corpus builds also
report bounded per-collection XML parsing progress, so long builds remain visibly
active. After parsing, the builder audits the complete normalized record graph
for duplicate database keys and broken relationships before opening SQLite. A
failed audit reports several conflicts together with source paths, so adapter
problems can be fixed in one pass. Pass `--verbose` (`-v`) to either command for
diagnostic logging.

Selecting `ddbdp` automatically fetches both `DDbDP/` and the linked
`HGV_meta_EpiDoc/` records. HGV is stored as documentary metadata, not as a
separate user-facing collection. The artifact is schema v4; an older artifact
is rejected with an actionable rebuild message.

To bundle semantic subject suggestions, install the semantic extra and point the builder at a
downloaded FastEmbed model snapshot for the pinned revision:

```bash
uv sync --extra semantic
hf download intfloat/multilingual-e5-small \
  --revision 4a4cddf9cf6d77a61cc1c73f824ec2127773db85 \
  --local-dir ./models/multilingual-e5-small
uv run papyrus-corpus-build dclp ddbdp translations \
  --semantic-model-dir ./models/multilingual-e5-small \
  --output ./data/papyrus-corpus
```

Add `--semantic-content` to the build command to also index text chunks and
document profiles for semantic discovery (it always requires
`--semantic-model-dir`; see [semantic retrieval](docs/semantic-retrieval.md) for
what it costs and what it adds). Omit it to keep subject-suggestion-only builds.

The builder stores normalized HGV subject labels, float32 vectors, the model
snapshot, and file digests in the artifact. Chat-time queries use the same
local model and fuse lexical vocabulary matches with dense ranking. Suggested
labels are then applied as exact HGV subject filters, so the assistant can
report both narrow and broader cohorts with exact counts, label prevalence,
and subject-annotation coverage. With content indexes, the chat and MCP tools
additionally expose `discover_documents` for ranked semantic candidates and
chunk-focused inspection; see the same page for the capability reporting and
rebuild rules.

`papyrus-chat` validates the artifact, binds to `127.0.0.1:8000`, and opens
the stock Pydantic AI chat UI. The UI provides persistent browser threads,
streaming responses, and visible tool activity. The application is local,
single-user, read-only, and does not maintain bespoke search or document
routes. Semantic suggestions are a planning aid; corpus counts and citations still
come only from exact local queries and inspections.

The configured endpoint must support reliable function/tool calling. A
plain-text-only completion endpoint cannot invoke corpus retrieval. Use the
model identifier exactly as the provider advertises it; identifiers may be
case-sensitive.

For a provider that implements both the OpenAI Responses API and its native
`web_search` tool, select the Responses transport with the
`openai-responses:` prefix and opt in to web search at startup:

```bash
export LLM_MODEL="openai-responses:exact-model-name"
uv run papyrus-chat --artifact ./data/papyrus-corpus --web-search
```

The prefix selects the Responses API transport; `--web-search` separately
enables the native tool. Native web search is optional and requires no
additional search API key. OpenAI compatibility alone does not guarantee that
an endpoint implements the Responses API or its native tool.

For an endpoint that supports Chat Completions but not native web search, omit
the prefix and install the provider-neutral DuckDuckGo historical-background tool:

```bash
uv sync --extra web
export LLM_MODEL="exact-model-name"
uv run papyrus-chat --artifact ./data/papyrus-corpus --web-search
```

Web search is disabled by default. When enabled, it can verify historical and
contextual background such as reign dates, chronology, Egyptian regnal-year
mechanics, terminology, institutions, and geography. Web results are cited as
web-sourced background and never replace local corpus evidence or contribute to
corpus counts; papyri records and transcriptions still come only from local
corpus tools.

### Research answer semantics

Every evidence-oriented answer is expected to include **Scope and method**:
the interpreted collections, inclusive date interval, transcription language,
and generated multilingual term groups. Candidate counts are exact for the
displayed structured filters, but are not exhaustive scholarly
classifications. Corpus documents are cited only with papyri.info URLs
returned by a corpus tool, and transcription evidence is kept distinct from
model-generated synthesis.

### Context management and research limits

Research continues until the model answers or you cancel. There is no default
application limit on model requests or compactions, and no automatic switch to an
“incomplete research” answer. Short questions can finish without compaction.

Before a model request, the agent estimates the context size, including the
question, history, instructions, and tool schemas. By default it compacts at
**65% of context capacity**, targeting **45%**. It keeps the question, a concise
narrative checkpoint, research notes and progress, and recent complete tool
exchanges that fit. Tool calls are never separated from their results. Original
tool arguments and results remain in a request-local evidence ledger, even when
they no longer fit in the prompt. The model can retrieve them using
`list_research_records` and `read_research_record`, including exact scoped counts,
quotations, and source references. Summaries and model-written notes are not new
citation sources.

Summarization uses the configured model and endpoint, without research tools.
The agent reduces or disables summary reasoning only where Pydantic AI's model
profile supports it; unknown compatible endpoints may offer no such control.
Summary text is requested to stay concise, independently of its generation
allowance. If a summary fails or cannot produce a usable checkpoint, the agent
retains its last valid narrative and builds a smaller context mechanically from
records and progress. It continues researching, with model-written summaries
disabled for the rest of that turn. Essential context that still cannot fit
produces an explicit error rather than silently rewriting the question.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LLM_CONTEXT_WINDOW` | Model registry capacity, otherwise 32,768 | Actual deployment capacity in tokens; minimum 4,096 |
| `LLM_MAX_TOKENS` | Unset: server default | Optional per-request generation limit, including reasoning and tool arguments |
| `PAPYRUS_SUMMARY_MAX_TOKENS` | Explicit `LLM_MAX_TOKENS`, otherwise server default | Optional summary generation override |
| `PAPYRUS_RESEARCH_REQUEST_LIMIT` | Unset: no limit | Optional total request limit per turn, including summaries, recovery, and citation repair; minimum 1 |
| `PAPYRUS_RUN_TIMEOUT_SECONDS` | Unset: no limit | Optional positive elapsed-time limit across research, summaries, and recovery |
| `PAPYRUS_RUN_COST_LIMIT_USD` | Unset: no limit | Optional positive estimated USD cost limit; requires supported model pricing metadata |
| `PAPYRUS_COMPACTION_LIMIT` | Deprecated and ignored | Emits a startup warning when set |

The old 16,384/8,192 generation defaults and 16-request/3-compaction defaults no
longer apply. With no generation override, the application omits the generation
setting and leaves space for output and estimation uncertainty. This reservation
**cannot guarantee compatibility with unknown server generation defaults**.
With an explicit generation limit, the agent reserves that full allowance plus
a safety margin of at least 512 tokens or 5% of context capacity. It compacts
earlier if needed, without silently lowering the configured generation limit.
Overrides must leave at least 1,024 input tokens plus the safety margin; the
question and tool instructions may require more.

Automatic capacity metadata can describe a model's maximum rather than the
window your deployment actually accepts. Token estimates use UTF-8 bytes
(including Greek text) and provider-reported input usage when available, and
are rebaselined after history changes. Set `LLM_CONTEXT_WINDOW` explicitly for a
local or proxied deployment whose actual window differs. For example:

```bash
export LLM_CONTEXT_WINDOW=65536
# Leave generation settings unset to use the server defaults.
uv run papyrus-chat --artifact ./data/papyrus-corpus
```

Summaries and recovery incur real latency and usage. Optional request, time,
and cost limits stop with a specific error; they do not force a partial answer.
Cost is estimated from available usage and registry prices, which may differ
from custom deployment pricing. Cost checks occur after responses and before
further requests, so the request that crosses a limit can still incur charges.
Unavailable pricing rejects a configured cost limit at startup. Increasing or
unsetting an explicit request limit permits more research but does not increase
context capacity or the server's per-request output allowance.

The stock browser UI retains the full transcript. Follow-ups reconstruct evidence,
progress, and saved notes from recognized tool exchanges in that transcript.
Each turn starts fresh counters and operational limits; independent chats do not
share evidence. There is no conversation database or persisted summary cache,
so follow-ups may need to compact the submitted history again. Legacy tool
results can supply evidence, but results without pagination metadata cannot
establish complete search coverage.

#### Pagination and measured coverage

`search_documents` and `discover_documents` accept `offset` (default `0`) and a
page size of 1–100. Follow `next_offset` until it is absent. Structured search
paginates after distinct-document ranking and retains exact candidate counts;
facet counts are independent of pagination. Semantic discovery traverses its
complete eligible indexed ranking, retaining reciprocal-rank fusion and stable
tie-breakers. Removing the former hidden 200-candidates-per-channel cutoff can
change rankings because previously discarded channel contributions now count.
Ranked-candidate totals, structural scope, and index coverage are separate
measurements. Semantic candidates are not verified thematic matches.

`inspect_documents` remains useful for focused excerpts. For sequential reading,
`read_document_passages` returns up to five source-ordered windows of at most
2,000 Unicode characters each. Windows include passage IDs, exact character
offsets, source references, and available line references. Continue with
`next_cursor`, which is bound to the document and artifact; oversized passages
continue before the next passage begins. This requires no artifact rebuild.
The new passage tool is exposed in chat, not through MCP; existing MCP tools
remain compatible with the additive query and result fields.

The model can use `get_research_progress` to check searches and
`update_research_notes` to retain objectives, interpretations, and remaining work.
Answers following corpus research receive a deterministic coverage note:

- **Unique candidate documents retrieved:** distinct documents returned by
  structured or semantic searches, counting overlaps once.
- **Documents with inspected excerpts:** documents whose source text was delivered
  by focused inspection or passage reading. This does not imply complete reading.
- **Documents with all stored passage text retrieved:** every stored passage's
  character range has been delivered, with overlapping windows counted once.
- **Result pages outstanding:** the number of searches with known unreturned
  candidates. Unknown pagination coverage is reported separately.

Coverage measures source material delivered for review. It does not prove the
model understood every passage, and completing a ranking does not establish
exhaustive thematic discovery. Interpretive uncertainty belongs in the answer,
without a blanket incompleteness warning.

#### Generation exhaustion and troubleshooting

Context compaction happens **between requests**. It cannot prevent a model from
using its entire generation allowance on thinking during one request. When a
response terminates because of length, the agent discards its answer text and
tool arguments before accepting or executing them. It tries that logical request
once more, non-streaming, with a compact checkpoint and reduced reasoning where
supported. The question, explicit generation settings, and current tool
permissions are preserved. A second exhaustion or provider failure is an error;
no truncated draft is accepted. This recovery is separate from the one tool-free
citation-repair attempt allowed for unsupported citations.

If generation exhaustion persists, adjust the server's generation/reasoning
settings, use an appropriate explicit `LLM_MAX_TOKENS`, or reduce reasoning
where the provider supports it. An unknown Qwen-compatible endpoint may not
support the generic reasoning control. Larger output allowances leave less
space for evidence. If context overflow persists, first check the startup log's
resolved capacity and set the correct `LLM_CONTEXT_WINDOW`; the fallback can
still exceed a smaller deployment's window.

Very large evidence collections can exceed a single answer's provider output
limit even when research succeeds. Request a narrower synthesis or use further
turns to explore portions of the evidence. Downloadable reports are separate
work. Compaction and one recovery attempt cannot guarantee completion during
provider outages.

Tool events stream immediately, while answer text is emitted only after citation
validation. Checkpoints and discarded answer drafts are kept out of the answer
stream. Cancellation stops research, summarization, and recovery without starting
another request. Context-management logs record capacity, context measurements,
compaction changes, recovery attempts, coverage counts, and stop reasons, without
logging prompts or evidence text.
