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

### Research answers

Answers explain which collections, dates, languages, and search terms were used.
Document claims cite links returned by the corpus tools. Search counts describe
matches to those filters; thematic relevance still requires interpretation.

A coverage note reports unique documents found, documents inspected through
excerpts, documents whose complete stored text was retrieved, and searches with
results still to retrieve. Overlapping searches and repeated reads count once.
Reading an excerpt does not count as reading a whole document, and completing a
result list does not guarantee that every relevant passage was found.

Broad requests such as "all evidence you can find" guide the agent to investigate
relevant aspects, inspect promising candidates, and synthesize when further searches
add little useful evidence. Unread search results describe retrieval coverage; they
do not require exhausting every semantic ranking, which can include weakly related
documents throughout the corpus. Explicit requests to enumerate every result within
defined filters still require paging through that inventory or reporting what remains.

### Context management and research limits

Research continues until the model answers or you cancel. By default, there is
no limit on the number of model requests or summaries.

When the conversation approaches 65% of the model's context window, the agent
summarizes earlier work, aiming to reduce it to 45%. The question and original
evidence remain available, including exact quotations, counts, and citations.
The checkpoint preserves research progress and recent decisions so the agent can
continue unresolved work or answer from existing findings. Saved notes distinguish
completed searches and rejected directions from questions that still need evidence.
Summaries use the same model and add time and usage costs. If summarization
fails, the agent continues with a shorter selection of saved material.
Follow-up questions use the browser's chat history and may need another summary.

The model server sets the response length unless you configure an override.
These optional environment settings apply to the chat agent:

| Setting                          | Default                                                 | Purpose                                                            |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
| `LLM_CONTEXT_WINDOW`             | Matching profile, then model metadata, otherwise 32,768 | Deployment's context capacity in tokens                            |
| `LLM_MAX_TOKENS`                 | Server default                                          | Tokens per response, including reasoning                           |
| `PAPYRUS_SUMMARY_MAX_TOKENS`     | `LLM_MAX_TOKENS`, if set; otherwise server default      | Tokens per summary response                                        |
| `PAPYRUS_RESEARCH_REQUEST_LIMIT` | No limit                                                | Total model requests per question, including summaries and retries |
| `PAPYRUS_RUN_TIMEOUT_SECONDS`    | No limit                                                | Total research time in seconds                                     |
| `PAPYRUS_RUN_COST_LIMIT_USD`     | No limit                                                | Estimated model cost per question in USD                           |

For settings specific to one deployment, copy
[`conf/model-profiles.example.toml`](conf/model-profiles.example.toml) to
`conf/model-profiles.toml` and enter your endpoint, model name, and server's
context capacity. The local file is gitignored; keep API keys in `.env`.
Use `PAPYRUS_MODEL_PROFILES` to select a file elsewhere.

Profiles apply only when both the endpoint and model match, including the
`openai-responses:` prefix for Responses models. Changing either uses the new
model's defaults unless another profile matches. An explicit `LLM_CONTEXT_WINDOW`
(or a policy supplied in Python) takes precedence over profile capacity.
Automatic capacity estimates may differ from what the server accepts.

The profile's `reasoning_adapter` controls summaries and recovery attempts:

- `auto` (default): use reasoning controls recognized by Pydantic AI. Summaries
  prefer thinking off; recovery uses low reasoning or retains a lower setting.
  Models without recognized support receive no added controls.
- `qwen-chat-template`: for compatible Chat Completions deployments, disable
  summary thinking with `chat_template_kwargs.enable_thinking` and use low
  reasoning for recovery. Select this only when your endpoint supports it.
- `none`: leave reasoning settings unchanged.

Normal research reasoning stays unchanged. These controls do not set a response
length, so generation limits remain the server's choice unless you override them.
An explicit response allowance leaves less room for evidence and can trigger
earlier summarization.

Reaching a configured limit stops the run with an error. Cost limits require
available model pricing; estimates may differ from your provider's charges, and
the response that crosses the limit can still be billed. Each follow-up question
starts fresh limits.

### Reading search results

Searches return pages of 1-100 documents. Both chat and MCP expose `offset` and
`next_offset` through `search_documents` and `discover_documents`, allowing the
agent to continue through the results. Semantic search can return the full
ranking of indexed candidates; these candidates still need inspection.

In chat, `inspect_documents` opens focused excerpts. `read_document_passages`
reads sequentially, returning up to five sections of 2,000 characters with source
and line references. The agent follows `next_cursor` to continue reading.
It can also recall earlier evidence and check research progress. Passage reading,
evidence memory, and automatic summarization are available in chat. Any MCP host
will manage its own conversation and memory.

### Troubleshooting long answers

Summarizing the conversation cannot prevent a model from spending its response
allowance on reasoning. If a response runs out of tokens, the agent discards it
and tries once more. Repeated failure produces an error without displaying an
unfinished answer. Reduce reasoning where your provider supports it, or increase
`LLM_MAX_TOKENS` within the server's limits.

For context overflow errors, check `LLM_CONTEXT_WINDOW` against your server's
configuration. Very broad questions may also produce more evidence than fits in
one answer; ask for a narrower synthesis or explore the findings over follow-up
questions. Provider outages can still interrupt research.
