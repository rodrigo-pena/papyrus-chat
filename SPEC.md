# Papyrus Chat proof of concept — specification

Status: Accepted
Last updated: 2026-08-31
Target: a local, reproducible proof of concept built from selected collections in [`papyri/idp.data`](https://github.com/papyri/idp.data)

The terms **MUST**, **SHOULD**, and **MAY** are normative.

## 1. Objective

Build a user-friendly local application that lets a papyrologist create a searchable corpus artifact from selected `idp.data` collections and converse with that corpus through an OpenAI-compatible language-model endpoint.

The proof of concept has two user-facing commands:

```console
uv run papyrus-corpus-build dclp translations --output ./papyrus-corpus
uv run papyrus-chat --artifact ./papyrus-corpus
```

The corpus builder MUST be deterministic, require no LLM or API credentials, record the exact upstream Git commit, and produce a self-contained local artifact. The chat application MUST ground answers in retrieved passages, show its evidence, and preserve the distinction between source text, published translation, and model-generated explanation.

This is a conventional retrieval-augmented generation (RAG) proof of concept. GraphRAG, embeddings, and a hosted corpus service are deliberately deferred until the simpler design has been evaluated with papyrologists.

## 2. Users and core journeys

### 2.1 Primary user

A papyrologist who can install `uv`, clone this project, and configure an LLM endpoint, but should not need to understand databases, embeddings, XML, or web-server administration.

### 2.2 Build a corpus

1. The user chooses one or more supported collections.
2. The builder obtains only the selected source directories, or uses a local Git checkout.
3. It resolves the requested Git ref to an exact commit.
4. It validates and extracts the selected EpiDoc XML records.
5. It creates a local artifact and reports its contents, provenance, size, and any errors.

### 2.3 Search and chat

1. The user starts the application with an artifact path.
2. A local web interface opens in the default browser.
3. The user searches by identifier, title, metadata, or passage text.
4. The user selects a document or asks a corpus-level question.
5. The application retrieves relevant evidence and sends only that evidence with the question to the configured LLM.
6. The answer links every substantive claim back to visible corpus evidence.

## 3. Scope

### 3.1 Collections supported in the proof of concept

| CLI name       | Upstream directory | Initial purpose                                  |
| -------------- | ------------------ | ------------------------------------------------ |
| `dclp`         | `DCLP/`            | Literary papyrus metadata and available editions |
| `translations` | `Translations/`    | Published translation records                    |

Collection names MUST be canonicalized to lowercase and matched case-insensitively. The CLI MUST use separate positional arguments, not a quoted comma-separated value:

```console
# Correct
uv run papyrus-corpus-build dclp translations

# Not part of the interface
uv run papyrus-corpus-build "DCLP, Translations"
```

DCLP and Translations are useful together for testing the pipeline, but MUST NOT be presented as a completely joined corpus. Translation records often concern documentary papyri rather than DCLP records. Links between records may be shown only when supported by explicit identifiers in the source.

Some DCLP records contain metadata but no edition text. These MUST remain discoverable and MUST be visibly labelled **metadata only**.

### 3.2 Explicit non-goals

The proof of concept does not include:

- APD, APIS, DDbDP, HGV, Biblio, Historical, RDF, or Validation collections;
- images, OCR, image-to-text alignment, or IIIF integration;
- embeddings, a vector database, knowledge-graph construction, or GraphRAG;
- lemmatization, morphological search, or specialized Ancient Greek query expansion;
- a hosted search API, MCP server, shared multi-user deployment, or authentication;
- PyPI publication or `uvx` installation;
- publication of prebuilt corpus artifacts;
- incremental artifact updates;
- editing or writing data back to `idp.data`;
- LLM installation, model serving, or provider account management;
- reproducible LLM wording. Retrieval and corpus contents are reproducible; model output is not guaranteed to be.

## 4. Capability map

The capabilities form one vertical product and share this specification, but their boundaries MUST remain independently testable.

| Capability         | Responsibility                                            | Depends on         |
| ------------------ | --------------------------------------------------------- | ------------------ |
| `artifact-format`  | Versioned manifest, SQLite schema, provenance, validation | —                  |
| `corpus-builder`   | Source acquisition, EpiDoc extraction, artifact creation  | `artifact-format`  |
| `corpus-retrieval` | Identifier lookup, full-text search, evidence packets     | `artifact-format`  |
| `chat-runtime`     | Provider configuration and evidence-grounded conversation | `corpus-retrieval` |
| `local-web-ui`     | Search, document view, chat, and evidence display         | `chat-runtime`     |

Recommended implementation order:

1. `artifact-format`
2. `corpus-builder`
3. `corpus-retrieval`
4. `chat-runtime`
5. `local-web-ui`

## 5. Technical design

### 5.1 Technology choices

- Python 3.12 or newer
- `uv` for environments, dependency locking, and command execution
- Typer for the two command-line interfaces
- `lxml` for safe and practical EpiDoc XML processing
- SQLite and FTS5 for the self-contained searchable artifact
- Pydantic for manifest and boundary models
- FastAPI, Uvicorn, and Jinja2 for the local server-rendered web application
- `httpx` for the OpenAI-compatible HTTP interface
- pytest, Ruff, and ty for tests, linting/formatting, and type checking

All exact dependency versions MUST be recorded in `uv.lock`. The browser UI SHOULD use server-rendered HTML and a small amount of bundled JavaScript. It MUST NOT require Node.js or fetch frontend assets from a CDN at runtime.

### 5.2 Project layout

The repository SHOULD use one Python distribution with two console entry points:

```text
.
├── SPEC.md
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── papyrus_chat/
│       ├── artifact/
│       │   ├── manifest.py
│       │   ├── schema.py
│       │   └── validation.py
│       ├── builder/
│       │   ├── cli.py
│       │   ├── source.py
│       │   ├── pipeline.py
│       │   └── collections/
│       │       ├── dclp.py
│       │       └── translations.py
│       ├── retrieval/
│       │   ├── identifiers.py
│       │   ├── search.py
│       │   └── evidence.py
│       ├── chat/
│       │   ├── cli.py
│       │   ├── provider.py
│       │   └── conversation.py
│       └── web/
│           ├── application.py
│           ├── templates/
│           └── static/
└── tests/
    ├── fixtures/idp.data/
    ├── artifact/
    ├── builder/
    ├── retrieval/
    ├── chat/
    └── integration/
```

The entry points in `pyproject.toml` MUST be equivalent to:

```toml
[project.scripts]
papyrus-corpus-build = "papyrus_chat.builder.cli:app"
papyrus-chat = "papyrus_chat.chat.cli:app"
```

## 6. Corpus-builder interface

### 6.1 Command

```console
uv run papyrus-corpus-build COLLECTION... [OPTIONS]
```

Supported options:

| Option                 | Default                                  | Behavior                                                 |
| ---------------------- | ---------------------------------------- | -------------------------------------------------------- |
| `COLLECTION...`        | required                                 | One or more of `dclp`, `translations`                    |
| `-o`, `--output PATH`  | `./papyrus-corpus`                       | Destination artifact directory                           |
| `--source URL_OR_PATH` | `https://github.com/papyri/idp.data.git` | Git URL or local Git checkout                            |
| `--ref GIT_REF`        | `master`                                 | Branch, tag, or commit to resolve and record             |
| `--force`              | false                                    | Explicitly allow replacement of the destination artifact |
| `--list-collections`   | —                                        | Print supported collection names and exit                |

If no collection is supplied, the command MUST fail with a concise message and show the supported names. It MUST NOT silently build every collection.

### 6.2 Source acquisition

For a remote source, the builder MUST:

- use a user cache outside the artifact;
- use Git partial-clone and sparse-checkout behavior so only selected collection blobs are downloaded;
- fetch or resolve `--ref` and record the resulting full commit SHA;
- read files in stable, lexicographically sorted path order;
- avoid modifying a user's existing Git checkout.

For a local source, the path MUST be a Git checkout and `--ref` MUST resolve to a commit in it. A dirty working tree MUST NOT change the build: files MUST be read from the resolved commit, not from uncommitted state.

The artifact MUST remain usable after the cache or source checkout is removed.

### 6.3 XML processing

The collection adapters MUST safely parse EpiDoc XML with external entity expansion and network resolution disabled.

For every record they MUST extract, when present:

- collection and stable document identifier;
- all useful source identifiers and identifier namespaces;
- title and descriptive metadata;
- declared languages;
- edition or translation passages;
- `textpart` structure and line or milestone references;
- source-relative XML path and a structural locator;
- scholarly uncertainty represented by elements such as `supplied`, `unclear`, `gap`, and `certainty`/`cert`.

The builder MUST keep two text representations:

- `display_text`: a faithful, readable rendering that retains visible uncertainty and editorial signals;
- `search_text`: a normalized representation used only for retrieval.

Normalization MUST NOT overwrite the display form. The proof of concept MAY omit unsupported EpiDoc structures from normalized search text, but MUST record a warning rather than silently invent content.

### 6.4 Failure and replacement behavior

The artifact MUST be assembled and validated in a temporary sibling directory, then moved into place atomically.

- If the output exists without `--force`, the command MUST fail before building.
- With `--force`, only the exact requested artifact directory may be replaced.
- If source acquisition, parsing, indexing, or validation fails, the previous artifact MUST remain intact.
- A malformed source record MUST be reported with its collection and path. The proof of concept MUST fail the build rather than publish a partial artifact.
- Secrets MUST never be required, read, or written during corpus construction.

### 6.5 Completion report

On success, the command MUST print:

- artifact path;
- selected collections;
- resolved source commit;
- document and passage counts;
- artifact size;
- logical content hash;
- elapsed time.

Progress SHOULD be visible during source acquisition, parsing, and indexing without overwhelming the terminal with one line per file.

## 7. Artifact contract

### 7.1 Directory structure

```text
papyrus-corpus/
├── manifest.json
├── corpus.sqlite
└── ATTRIBUTION.md
```

No source checkout, Python environment, API key, or model cache may be embedded in the artifact.

### 7.2 Manifest

`manifest.json` MUST be UTF-8 JSON and contain at least:

```json
{
    "artifact_schema_version": 1,
    "builder": {
        "name": "papyrus-corpus-build",
        "version": "0.1.0"
    },
    "source": {
        "url": "https://github.com/papyri/idp.data.git",
        "requested_ref": "master",
        "resolved_commit": "FULL_GIT_SHA"
    },
    "collections": ["dclp", "translations"],
    "statistics": {
        "documents": 0,
        "passages": 0,
        "parse_errors": 0
    },
    "logical_content_hash": "sha256:...",
    "created_at": "RFC_3339_TIMESTAMP"
}
```

Collections MUST be sorted canonically. The logical content hash MUST be calculated from a documented canonical representation of source commit, build-relevant options, documents, identifiers, and passages. It MUST exclude volatile values such as timestamps and SQLite page layout. Identical inputs and builder versions MUST produce the same logical content hash; byte-identical SQLite files are not required.

The chat application MUST reject an unsupported artifact schema major version with an actionable message.

### 7.3 SQLite logical schema

The physical schema may evolve during implementation, but schema version 1 MUST expose these concepts:

- `documents`: stable ID, collection, title, languages, metadata, source path, structural source information, and canonical upstream URL where possible;
- `identifiers`: document ID, namespace, and identifier value, indexed for exact lookup;
- `passages`: stable passage ID, document ID, kind (`edition` or `translation`), sequence, textpart/line references, display text, search text, and uncertainty metadata;
- `passages_fts`: FTS5 index over appropriate passage and document search fields.

Foreign keys MUST be enabled and an artifact MUST pass SQLite integrity and foreign-key checks before publication.

Stable internal IDs MUST derive from collection, source identity, and structural location rather than insertion order.

### 7.4 Provenance and attribution

Every displayed passage MUST be traceable to:

- its collection;
- its source document identifier;
- the exact `idp.data` commit;
- the source-relative XML path;
- a textpart, line range, or structural locator when available.

`ATTRIBUTION.md` MUST state that `idp.data` is distributed under CC BY 3.0, link to the upstream repository and its README, name the selected contributing projects when known, record the commit, and explain that model-generated prose is not part of the source corpus.

## 8. Retrieval contract

Retrieval MUST be local and deterministic for a fixed artifact and query.

The application MUST attempt, in order:

1. normalized exact identifier lookup;
2. SQLite FTS5 search over passages and useful document fields;
3. deterministic tie-breaking by score, collection, document ID, and passage sequence.

Retrieval MUST support:

- corpus-wide search;
- search restricted to a selected document;
- filtering by collection and passage kind;
- metadata-only results when no passage text exists.

The retrieval layer MUST return an **evidence packet**, not an unstructured text blob. Each evidence item MUST carry display text, document metadata, passage kind, source locator, and a human-readable citation label.

The initial ranking MAY use a fixed FTS5/BM25 configuration. Its parameters MUST be explicit in code and covered by tests. Embeddings are not a fallback in this proof of concept.

## 9. Chat application

### 9.1 Command

```console
uv run papyrus-chat --artifact PATH [OPTIONS]
```

Options:

| Option            | Default     | Behavior                        |
| ----------------- | ----------- | ------------------------------- |
| `--artifact PATH` | required    | Corpus artifact directory       |
| `--host HOST`     | `127.0.0.1` | Local bind address              |
| `--port PORT`     | `8000`      | Local HTTP port                 |
| `--no-open`       | false       | Do not open the default browser |

The application MUST validate the manifest, schema compatibility, required files, SQLite integrity, and provider configuration before opening the browser.

### 9.2 LLM configuration

The proof of concept supports an OpenAI-compatible Chat Completions endpoint configured through:

```console
export LLM_BASE_URL="https://provider.example/v1"
export LLM_API_KEY="..."
export LLM_MODEL="model-name"
uv run papyrus-chat --artifact ./papyrus-corpus
```

Contract:

- `LLM_BASE_URL` is required and identifies the API root; a trailing slash is allowed.
- `LLM_MODEL` is required.
- `LLM_API_KEY` is optional so an unauthenticated local server can be used.
- The application calls `chat/completions` relative to the API root.
- No provider-specific SDK behavior is required in the proof of concept.
- The API key MUST stay on the Python server. It MUST NOT appear in HTML, browser JavaScript, logs, artifacts, or error pages.

The interface MUST give clear guidance when configuration is missing, the endpoint cannot be reached, authentication fails, or the model rejects the request.

### 9.3 Grounding behavior

For each question, the application MUST:

1. retrieve a bounded set of evidence locally;
2. send only the question, conversation context needed for coherence, and retrieved evidence to the provider;
3. instruct the model to distinguish corpus statements from inference;
4. require inline evidence markers that the application can map back to evidence items;
5. display the answer together with the exact evidence supplied to the model.

If there is insufficient evidence, the answer SHOULD say so and suggest a narrower search. It MUST NOT invent a papyrus text for metadata-only records.

Published translations MUST be labelled as source translations. Any new translation, summary, restoration, or interpretation produced by the model MUST be labelled as model-generated.

Corpus text, metadata, retrieved passages, and model output MUST be treated as untrusted content. They MUST never be interpreted as instructions to run commands, access files, or reveal secrets.

## 10. Web-interface requirements

The first interface MUST contain:

- a prominent search field;
- collection and passage-kind filters;
- a results list with identifier, title, collection, passage kind, and a short snippet;
- a selected-document view showing metadata and available passages;
- a chat panel that can operate over the selected document or the corpus;
- an expandable **Evidence used** section for every answer;
- copyable citations containing source identifier and locator;
- visible **metadata only**, **source translation**, and **model-generated** labels where applicable.

The UI MUST use terminology understandable without RAG knowledge. It SHOULD say "Search the corpus" and "Evidence used," not "vector retrieval," "chunks," or "context window."

The interface MUST:

- work with keyboard navigation;
- use semantic HTML and associated form labels;
- provide sufficient color contrast without relying on color alone;
- render polytonic Greek and common papyrological symbols legibly;
- remain usable on a laptop-sized display and a narrow browser window;
- escape corpus and model content before rendering it as HTML;
- bind only to `127.0.0.1` by default;
- contain no telemetry or third-party browser requests.

## 11. Code style

Code MUST use complete type annotations at package boundaries, `pathlib.Path` for filesystem paths, small cohesive modules, and explicit immutable data models for values passed between parsing, storage, retrieval, and chat. Prefer descriptive public subpackages over large modules split into underscore-prefixed files.

Representative style:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict


class SourceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository_url: str
    commit: str
    path: str
    locator: str | None = None


class Passage(BaseModel):
    model_config = ConfigDict(frozen=True)

    passage_id: str
    document_id: str
    kind: Literal["edition", "translation"]
    display_text: str
    search_text: str
    source: SourceReference
```

Errors at CLI and HTTP boundaries MUST be converted to concise, actionable user messages. Internal exceptions and source paths MAY be included under a verbose/debug mode later, but tracebacks MUST NOT be the default user experience.

## 12. Testing strategy

### 12.1 Fixtures

Commit a small, licensing-compatible set of representative XML fixtures and record their upstream commit and paths. Fixtures MUST cover:

- a DCLP record with edition text;
- a DCLP metadata-only record;
- a translation with textparts and line/milestone structure;
- supplied, unclear, gap, and certainty markup;
- multiple identifier namespaces;
- Unicode/polytonic Greek;
- deliberately malformed XML created by the project for failure testing.

### 12.2 Automated tests

Tests MUST include:

- unit tests for each collection adapter and normalization rule;
- manifest and schema compatibility tests;
- stable-ID and logical-hash reproducibility tests;
- SQLite integrity, foreign-key, and FTS behavior tests;
- CLI tests for invalid collections, output replacement, source refs, and successful builds;
- retrieval tests for exact identifiers, ranking, filters, and metadata-only documents;
- provider tests using a mock HTTP server, never a real paid endpoint;
- prompt/response tests proving evidence markers map to displayed citations;
- web tests for escaping, validation errors, core routes, and accessible labels;
- an integration test that builds a fixture artifact and queries it through the application.

The default test suite MUST NOT clone the live upstream repository or call an external LLM. Optional network smoke tests MUST be explicitly marked and excluded by default.

### 12.3 Quality commands

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

All four commands MUST pass before a proof-of-concept release is tagged.

## 13. Performance and usability targets

These targets are initial and MUST be measured on a documented reference machine:

- first visible progress within 2 seconds of command startup;
- warm-cache build of DCLP plus Translations in no more than 5 minutes on a modern four-core laptop;
- first remote build SHOULD complete within 15 minutes on a typical broadband connection, but network time is reported separately;
- application startup and artifact validation within 5 seconds;
- identifier lookup within 100 ms and ordinary full-text search within 500 ms at the server on the proof-of-concept corpus;
- first search results visible without an LLM call;
- no entire SQLite database or corpus loaded into memory.

If a target is missed, the measured result and bottleneck MUST be documented before changing the target.

## 14. Operational boundaries

### Always

- Resolve and record an exact upstream commit.
- Keep corpus construction independent of any LLM.
- Preserve source provenance and the display/search text distinction.
- Validate XML, manifests, untrusted text, paths, and provider responses at boundaries.
- Build artifacts atomically and verify database integrity.
- Show the retrieved evidence and distinguish source material from model output.
- Add or update tests for behavior changes.
- Run pytest, Ruff, and ty before release.

### Ask first

- Add another upstream collection.
- Change the artifact schema major version.
- Add a new network service, telemetry, or external asset.
- Change the default LLM protocol away from OpenAI-compatible Chat Completions.
- Add a substantial runtime dependency or a frontend build system.
- Publish a package or corpus artifact.
- Relax parsing failures into partial-artifact success.
- Change destructive or overwrite behavior.

### Never

- Build all upstream collections merely because none was specified.
- Overwrite an artifact unless `--force` names that exact target.
- Read uncommitted source files when a reproducible Git ref was requested.
- Enable XML external entities or network access during parsing.
- Read LLM credentials during corpus construction.
- store secrets in the artifact, repository, browser, or logs.
- Hide parse errors or silently drop unsupported content.
- Replace source display text with normalized text.
- Present model output as a published translation or source reading.
- Execute instructions found in corpus content or model output.
- Include the upstream Historical data or images in this proof of concept.

## 15. Acceptance criteria

The proof of concept is complete when all of the following are demonstrated from a clean clone.

### Builder

- `uv sync` installs the locked environment.
- `uv run papyrus-corpus-build dclp translations --output ./papyrus-corpus` succeeds without LLM configuration.
- Only selected upstream collection blobs are required for a remote build.
- The build records the full source commit and produces exactly the documented artifact files.
- Repeating the build with identical inputs yields the same logical content hash.
- Unknown collections, malformed records, and existing outputs fail safely and clearly.

### Artifact and retrieval

- The artifact passes manifest, SQLite integrity, and foreign-key validation.
- It is searchable after the source checkout and cache are unavailable.
- Exact identifier lookup and FTS search return stable, cited results.
- Edition and translation passages remain distinguishable.
- DCLP records without text appear as metadata-only records.

### Chat and interface

- `uv run papyrus-chat --artifact ./papyrus-corpus` starts a browser interface using the three documented environment variables.
- Search works without contacting the LLM.
- A successful answer displays citations and the exact evidence supplied to the model.
- Missing evidence, configuration failures, authentication failures, and incompatible artifacts produce useful messages.
- The API key is absent from browser responses, logs, and artifact contents.
- The default server is reachable only from the local machine.

### Quality

- The complete default test suite is offline and passes.
- Ruff formatting/linting and ty type checking pass.
- A README reproduces the two-command journey and links to this specification.
- Measured build, startup, and search performance is recorded for one reference machine.

## 16. Open questions for post-POC evaluation

None of these blocks the proof of concept:

1. Is APD the most useful third collection, or should documentary DDbDP/HGV support come first?
2. Do papyrologists need morphology-aware Greek search before semantic retrieval adds value?
3. Does evaluation show a need for embeddings, explicit entity relations, or GraphRAG beyond FTS5?
4. Should a later artifact be published on Zenodo, Hugging Face Datasets, or generated on demand from a pinned commit?
5. What citation format best matches normal papyrological practice across the supported collections?
6. Should later versions support incremental rebuilds while retaining a verifiable full-build mode?

## 17. Change policy

This specification is a living contract. Implementation discoveries MAY refine details, but changes to scope, public commands, artifact compatibility, failure behavior, security boundaries, or acceptance criteria MUST update this file in the same change and be called out explicitly during review.
