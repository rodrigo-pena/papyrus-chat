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

The chat agent automatically summarizes long research runs using the same model
and endpoint. It preserves the current question and keeps exact tool evidence
separate from the narrative summary. Counts retain their search scope. Inspected
excerpts retain document IDs, line references, and citation URLs. Whole records
that cannot fit are omitted with an explicit notice in the model's context.

Each user turn allows, by default,
**16 model requests, including at most 3 summaries**, plus
**2 reserved final-answer attempts**. Once research reaches its budget, or
compaction cannot reclaim enough space, corpus and web tools are disabled and
the agent answers from retained evidence. Such answers include an incomplete
research notice. Short questions can finish sooner without any summarization.
The second final-answer attempt is reserved for validation repair.

| Environment variable             | Default                                   | Purpose                                                               |
| -------------------------------- | ----------------------------------------- | --------------------------------------------------------------------- |
| `LLM_CONTEXT_WINDOW`             | Model registry capacity, otherwise 32,768 | Override with the actual deployment capacity in tokens; minimum 4,096 |
| `LLM_MAX_TOKENS`                 | Smaller of 16,384 or 50% of capacity       | Generation limit per research or final-answer request, including reasoning |
| `PAPYRUS_SUMMARY_MAX_TOKENS`     | Smaller of 8,192 or 25% of capacity        | Generation limit per summary request, including reasoning             |
| `PAPYRUS_RESEARCH_REQUEST_LIMIT` | `16`                                      | Requests available for research and summaries; minimum 1              |
| `PAPYRUS_COMPACTION_LIMIT`       | `3`                                       | Maximum summary attempts within the research budget; minimum 0        |

Automatic model metadata may describe a provider's maximum rather than your
deployment's configured window. For local or proxied models, set
`LLM_CONTEXT_WINDOW` to the capacity your server actually accepts. For example,
for a deployment configured to accept 65,536 tokens:

```bash
export LLM_CONTEXT_WINDOW=65536
uv run papyrus-chat --artifact ./data/papyrus-corpus
```

Generation limits cover reasoning tokens, tool-call arguments, and visible text.
Requests reserve the full generation allowance plus a safety margin (the larger
of 512 tokens or 5% of capacity). Compaction starts at 65% of estimated capacity
or the remaining input allowance, whichever is smaller. It targets the smaller
of 45% of capacity or 75% of that input allowance. Summaries reserve their own
generation allowance and keep saved checkpoint text short: at most the smaller
of 2,048 tokens or 12.5% of capacity, allowing for estimation uncertainty.
Estimates account for UTF-8 text, instructions, and tool schemas, using reported
input usage when available. Startup logs report the resolved context capacity
and generation limits. Overrides must leave at least 1,024 input tokens plus the
safety margin; an individual question may need more space.

Summaries incur real model latency and usage, and consume research request slots.
Increasing the research budget permits more work but does not increase context
capacity. A follow-up question starts a fresh budget. The browser retains its
full transcript. There is no server conversation database or summary cache, so
follow-ups may need to summarize submitted history again. Evidence and counters
are isolated between chat requests.

If an endpoint still reports a context overflow, verify its capacity and set the
explicit override; an unknown model's 32,768-token fallback can still be too
large. A current question that cannot fit with required instructions and output
space is rejected. If a summary fails, the agent attempts a partial answer using
its last valid checkpoint and bounded original evidence. Provider outages or
exhausted answer-validation attempts still produce an error; compaction cannot
guarantee completion in those cases. Cancellation stops the run without starting
a final answer.

If the UI reports **"Model token limit (...) exceeded before any response was
generated"**, the model exhausted a single request's generation allowance,
possibly while thinking. Increase `LLM_MAX_TOKENS` within your deployment's
output and context limits, or select a lower reasoning effort in the UI if the
provider supports it. `PAPYRUS_SUMMARY_MAX_TOKENS` independently controls summary
generation. For example, with a deployment supporting a 65,536-token context
and these output limits:

```bash
export LLM_CONTEXT_WINDOW=65536
export LLM_MAX_TOKENS=32768
export PAPYRUS_SUMMARY_MAX_TOKENS=16384
uv run papyrus-chat --artifact ./data/papyrus-corpus
```

Larger generation limits can increase latency and usage, and leave less room
for evidence. Increasing the research request count does not fix a generation
limit error. Compaction happens between requests and cannot interrupt thinking
within a request, so a model can still exhaust its generation allowance.

The server logs resolved capacity, summary attempts, compaction size changes,
finalization reasons, and failures. Context-management logs contain only
identifiers and measurements. **These controls apply to the Pydantic AI chat agent**;
they do not change corpus retrieval or the MCP interface.

### Builder options

```bash
uv run papyrus-corpus-build COLLECTION... [OPTIONS]

# COLLECTION...   one or more of: dclp, ddbdp, translations (case-insensitive)
# -o, --output    destination directory (default ./data/papyrus-corpus)
# --source        Git URL (default upstream) or a local idp.data Git checkout
# --ref           branch, tag, or commit to build from (default master)
# --force         replace an existing artifact at exactly the given path
# --semantic-model-dir  local FastEmbed model snapshot to bundle for subject suggestions
# --semantic-content     also build chunk/profile indexes for semantic discovery
#                        (requires --semantic-model-dir); see docs/semantic-retrieval.md
# --list-collections
# -v, --verbose   include detailed diagnostic logging
```

Remote builds use a Git partial clone, sparse checkout, and a persistent Git
object reader, so only the selected source data is downloaded and records are
read from the resolved commit without launching Git once per XML file. Builds
are deterministic: identical inputs produce the same logical content hash in
`manifest.json` and the completion report. Reference measurements and known
bottlenecks are recorded in [docs/performance.md](docs/performance.md).

### Sample questions

- How many Greek texts are lists related to tax payments from the Islamic period (from the Arab conquest of Egypt)?
- Within this corpus, can you find lists structured by month? Make sure you know the names of Egyptian months used in this period.
- Can you summarize the kinds/categories of taxes attested in these documents?
- How were taxes collected in the Early Arab period in Egypt, based on the Greek papyri in the corpus?

## Development

```bash
uv run pytest                    # offline test suite (network tests excluded)
uv run pytest -m network         # optional smoke test against the real upstream
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

For a browser-only demo with a collaborator, see the [authenticated ngrok
sharing guide](docs/demo-for-others.md).

Test fixtures are pinned to an upstream commit with recorded provenance
([tests/fixtures/idp.data/PROVENANCE.md](tests/fixtures/idp.data/PROVENANCE.md));
`scripts/refresh_fixtures.py` lets an informed user re-pin them to HEAD of
`master` or an arbitrary commit.
