# papyrus-chat

Build a searchable, provenance-preserving corpus from
[papyri/idp.data](https://github.com/papyri/idp.data), then investigate it in
a local Pydantic AI chat. The assistant discloses its search scope, separates
local transcription evidence from model background, and links cited records to
papyri.info.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git
- An OpenAI-compatible endpoint with function/tool calling and streaming
  support. Set its base URL and model with `LLM_BASE_URL` and `LLM_MODEL`; an
  API key is optional for unauthenticated local endpoints.
- Internet access the first time the browser UI is opened, so Pydantic AI can
  fetch and cache its stock chat UI from the CDN.

## Quick start

```console
# 1. Install the locked environment
uv sync

# 2. Build the full selected documentary corpus (no LLM or credentials needed)
uv run papyrus-corpus-build dclp ddbdp translations --output ./papyrus-corpus

# 3. Configure the model and start the local chat
export LLM_BASE_URL="https://provider.example/v1"
export LLM_MODEL="model-name"
export LLM_API_KEY="..."   # optional for local, unauthenticated servers
uv run papyrus-chat --artifact ./papyrus-corpus
```

Both commands write timestamped stage logs to the terminal. Corpus builds also
report bounded per-collection XML parsing progress, so long builds remain visibly
active. Pass `--verbose` (`-v`) to either command for diagnostic logging.

Selecting `ddbdp` automatically fetches both `DDbDP/` and the linked
`HGV_meta_EpiDoc/` records. HGV is stored as documentary metadata, not as a
separate user-facing collection. The artifact is schema v2; an older artifact
is rejected with an actionable rebuild message.

`papyrus-chat` validates the artifact, binds to `127.0.0.1:8000`, and opens
the stock Pydantic AI chat UI. The UI provides persistent browser threads,
streaming responses, and visible tool activity. The application is local,
single-user, read-only, and does not maintain bespoke search or document
routes; research happens through the assistant's four corpus tools.

The configured endpoint must support reliable function/tool calling. A
plain-text-only completion endpoint cannot invoke corpus retrieval. For a
provider that supports Pydantic AI's OpenAI Responses native web search, use
the `openai-responses:` model prefix:

```console
export LLM_MODEL="openai-responses:model-name"
```

Native web search is optional and requires no additional search API key. When
it is unavailable, the assistant may use model knowledge for terminology or
historical context, but must label that material as model-supplied background;
web results never replace local corpus evidence.

### Research answer semantics

Every evidence-oriented answer is expected to include **Scope and method**:
the interpreted collections, inclusive date interval, transcription language,
and generated multilingual term groups. Candidate counts are exact for the
displayed structured filters, but are not exhaustive scholarly
classifications. Corpus documents are cited only with papyri.info URLs
returned by a corpus tool, and transcription evidence is kept distinct from
model-generated synthesis.

### Builder options

```console
uv run papyrus-corpus-build COLLECTION... [OPTIONS]

# COLLECTION...   one or more of: dclp, ddbdp, translations (case-insensitive)
# -o, --output    destination directory (default ./papyrus-corpus)
# --source        Git URL (default upstream) or a local idp.data Git checkout
# --ref           branch, tag, or commit to build from (default master)
# --force         replace an existing artifact at exactly the given path
# --list-collections
# -v, --verbose   include detailed diagnostic logging
```

Remote builds use a Git partial clone and sparse checkout, so only the
selected source data is downloaded. Builds are deterministic: identical
inputs produce the same logical content hash in `manifest.json` and the
completion report. Reference measurements and known bottlenecks are recorded
in [docs/performance.md](docs/performance.md).

### Collaborator walkthrough

After starting the app, use one thread for this four-query smoke run:

1. "Find Greek documentary evidence about a Claudius-era money dispute. Show
   the date interpretation, generated Greek/German terms, exact candidate
   count, and linked papyri."
2. "Narrow that prior candidate set to the linked date interval. Which
   Egyptian month names are relevant?"
3. "Group the tax or payment categories in the results and give one linked
   papyrus example for each."
4. "Which statements are directly attested in the transcription, and which
   are broader historical inference or model background?"

Check that the first answer explains its scope, follow-up turns reuse and
refine the prior query, and every named corpus record opens at papyri.info.

## Development

```console
uv run pytest                    # offline test suite (network tests excluded)
uv run pytest -m network         # optional smoke test against the real upstream
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Test fixtures are pinned to an upstream commit with recorded provenance
([tests/fixtures/idp.data/PROVENANCE.md](tests/fixtures/idp.data/PROVENANCE.md));
`scripts/refresh_fixtures.py` lets an informed user re-pin them to HEAD of
`master` or an arbitrary commit.
