# papyrus-chat

Build a searchable corpus from selected [papyri/idp.data](https://github.com/papyri/idp.data)
collections and question it in your browser, with every answer grounded in
visible corpus evidence. A local proof of concept for papyrologists — see
[SPEC.md](SPEC.md) for the full specification.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git
- An OpenAI-compatible chat-completions endpoint (base URL + model name; an
  API key only if your endpoint requires one)

## Quick start

```console
# 1. Install the locked environment
uv sync

# 2. Build a corpus artifact (no LLM or credentials involved)
uv run papyrus-corpus-build dclp translations --output ./papyrus-corpus

# 3. Ask the corpus questions
export LLM_BASE_URL="https://provider.example/v1"
export LLM_MODEL="model-name"
export LLM_API_KEY="..."   # optional for local, unauthenticated servers
uv run papyrus-chat --artifact ./papyrus-corpus
```

`papyrus-chat` validates the artifact and your configuration, then opens
your browser on `http://127.0.0.1:8000`. Search works without any LLM;
answers always show the **Evidence used** and are labelled as
model-generated.

### Builder options

```console
uv run papyrus-corpus-build COLLECTION... [OPTIONS]

# COLLECTION...   one or more of: dclp, translations (case-insensitive)
# -o, --output    destination directory (default ./papyrus-corpus)
# --source        Git URL (default upstream) or a local idp.data Git checkout
# --ref           branch, tag, or commit to build from (default master)
# --force         replace an existing artifact at exactly the given path
# --list-collections
```

Remote builds use a Git partial clone and sparse checkout, so only the
selected collections' data is downloaded. Builds are deterministic:
identical inputs produce the same logical content hash (recorded in
`manifest.json` and the completion report). Measured performance for one
reference machine is recorded in [docs/performance.md](docs/performance.md).

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
