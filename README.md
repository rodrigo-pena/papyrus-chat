# Papyrus Chat

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

## Quick start

```console
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
# -o, --output    destination directory (default ./data/papyrus-corpus)
# --source        Git URL (default upstream) or a local idp.data Git checkout
# --ref           branch, tag, or commit to build from (default master)
# --force         replace an existing artifact at exactly the given path
# --list-collections
# -v, --verbose   include detailed diagnostic logging
```

Remote builds use a Git partial clone, sparse checkout, and a persistent Git
object reader, so only the selected source data is downloaded and records are
read from the resolved commit without launching Git once per XML file. Builds
are deterministic: identical inputs produce the same logical content hash in
`manifest.json` and the completion report. Reference measurements and known
bottlenecks are recorded in [docs/performance.md](docs/performance.md).

### Sample walkthrough

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
