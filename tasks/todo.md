# Papyrus Chat — task list

Companion to `tasks/plan.md`. Work tasks in order unless using the parallelization
notes in the plan. Mark checkboxes as you go; all four quality commands
(`uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`)
must pass at every task boundary unless a task's Verification section says otherwise.

---

## Phase 1: Foundation

### Task 1: Project skeleton, tooling, and stub CLIs

**Description:** Create the uv-managed Python project per SPEC §5: `pyproject.toml`
(requires-python ≥ 3.12, runtime deps typer/lxml/pydantic/fastapi/uvicorn/jinja2/httpx/platformdirs,
dev deps pytest/ruff/ty, both console entry points from SPEC §5.2), the
`src/papyrus_chat/` package with the five subpackages, Ruff + pytest configuration
(including a registered `network` marker, excluded by default), and stub Typer CLIs:
`papyrus-corpus-build` accepts `--list-collections` and fails with supported names when
no collection is given; `papyrus-chat --help` prints guidance that the real interface
arrives in Task 16.

**Acceptance criteria:**
- [ ] `uv sync` installs the locked environment and creates `uv.lock`
- [ ] `uv run papyrus-corpus-build --list-collections` prints `dclp` and `translations`
- [ ] `uv run papyrus-corpus-build` with no collection exits non-zero with a concise
      message naming the supported collections (SPEC §6.1)
- [ ] `uv run papyrus-chat --help` exits zero
- [ ] `network` pytest marker registered; default run excludes network tests
- [ ] All four quality commands pass on the skeleton

**Verification:**
- [ ] Tests pass: `uv run pytest`
- [ ] Build succeeds: `uv sync && uv run papyrus-corpus-build --list-collections`
- [ ] Manual check: `uv run papyrus-chat --help` prints usage; `uv run ruff format .` is clean

**Dependencies:** None

**Files likely touched:**
- `pyproject.toml`
- `src/papyrus_chat/__init__.py`
- `src/papyrus_chat/builder/cli.py`
- `src/papyrus_chat/chat/cli.py`
- subpackage `__init__.py` files, `tests/conftest.py`

**Estimated scope:** Medium (bootstrap exception: many files, mostly trivial `__init__.py`)

---

### Task 2: EpiDoc XML fixtures with provenance

**Description:** Commit a small, licensing-compatible fixture corpus under
`tests/fixtures/idp.data/` mirroring upstream layout (`DCLP/<prefix>/<tm>.xml`,
`Translations/<prefix>/<id>-<seq>.xml`), covering every category in SPEC §12.1:
DCLP record with edition text; DCLP metadata-only record (e.g., real-world
`DCLP/23/23702.xml` shape with an empty `edition` div); translation with textpart and
line/milestone structure; `supplied`/`unclear`/`gap`/`certainty` markup; multiple
identifier namespaces (dclp/TM/LDAB/dclp-hybrid; apisid/controlNo/ddbdp/ddb-hybrid/HGV);
polytonic Greek; and deliberately malformed XML created by this project. Record the
upstream commit and exact source paths in `tests/fixtures/idp.data/PROVENANCE.md`
with CC BY 3.0 attribution. Provide a documented refresh mechanism (small script,
e.g. `scripts/refresh_fixtures.py`) that regenerates the fixture set from an arbitrary
upstream ref — the pinned SHA by default, or HEAD of the upstream default branch
(`master`) / any chosen commit SHA — updating `PROVENANCE.md` accordingly (decided
2026-08-31). Refresh requires network and is never part of the default suite.

**Acceptance criteria:**
- [ ] Fixture set covers all seven SPEC §12.1 categories
- [ ] `PROVENANCE.md` records upstream commit SHA, source paths, and CC BY 3.0 attribution
- [ ] Refresh mechanism regenerates fixtures from the pinned SHA or another ref/SHA and
      updates `PROVENANCE.md`; network-dependent, excluded from the default suite
- [ ] Malformed fixture is clearly project-created and separated from upstream-derived files
- [ ] A smoke test enumerates fixtures and asserts each expected category is present

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder -k fixtures`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: fixture tree matches upstream naming conventions (see plan's
      "Upstream data reference")

**Dependencies:** None (parallel with Task 1; one-time network access to fetch samples
at the pinned commit — plan resolved decision 5)

**Files likely touched:**
- `tests/fixtures/idp.data/**` (data files)
- `tests/fixtures/idp.data/PROVENANCE.md`
- `scripts/refresh_fixtures.py`
- `tests/builder/test_fixture_inventory.py`

**Estimated scope:** Small (data, not code; bootstrap exception on file count)

---

## Checkpoint A: After Tasks 1–2

- [ ] All four quality commands pass
- [ ] `uv sync` + both CLI entry points work from a clean clone
- [ ] Fixture inventory test passes; provenance recorded
- [ ] Proceed to Phase 2

---

## Phase 2: Artifact format

### Task 3: Manifest models and artifact validation

**Description:** Implement `artifact/manifest.py` (frozen Pydantic models for the SPEC
§7.2 manifest: builder info, source info with requested ref + resolved commit, canonical
collections, statistics, logical-content-hash, created-at; UTF-8 JSON load/save) and
`artifact/validation.py` (validate an artifact directory: manifest present and parseable,
required files `manifest.json`/`corpus.sqlite`/`ATTRIBUTION.md`, supported schema major
version — rejecting incompatible majors with an actionable message per SPEC §7.2 — SQLite
`PRAGMA integrity_check` and `foreign_key_check`). Errors are concise user-facing
messages, not tracebacks (SPEC §11).

**Acceptance criteria:**
- [ ] Manifest round-trips through JSON with all SPEC §7.2 fields; collections sorted canonically
- [ ] Loading an unsupported `artifact_schema_version` major raises/returns an
      actionable compatibility error naming the supported version
- [ ] Validation detects: missing manifest, missing files, corrupt SQLite, foreign-key violations
- [ ] Validation of a well-formed minimal artifact passes

**Verification:**
- [ ] Tests pass: `uv run pytest tests/artifact`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: n/a (library-level task)

**Dependencies:** Task 1

**Files likely touched:**
- `src/papyrus_chat/artifact/manifest.py`
- `src/papyrus_chat/artifact/validation.py`
- `tests/artifact/test_manifest.py`
- `tests/artifact/test_validation.py`

**Estimated scope:** Medium

---

### Task 4: SQLite schema, record models, stable IDs

**Description:** Implement `artifact/records.py` (frozen boundary models:
`DocumentRecord`, `IdentifierRecord`, `PassageRecord` with edition/translation kind,
display/search text, `SourceReference` per SPEC §11) and `artifact/schema.py` (SQLite
DDL for `documents`, `identifiers` (indexed for exact lookup), `passages`,
`passages_fts` FTS5 index per SPEC §7.3; a writer that bulk-inserts records with foreign
keys enforced; a reader for documents/passages/identifiers; stable-ID derivation from
collection + source path (+ structural locator for passages), never insertion order).
Fail fast with an actionable message if the runtime SQLite lacks FTS5.

**Acceptance criteria:**
- [ ] Schema v1 exposes the four SPEC §7.3 concepts with the documented columns
- [ ] Writer inserts documents/identifiers/passages; FTS5 index contains search text
- [ ] Foreign keys enforced; orphan passages rejected
- [ ] Stable IDs are identical across two separate write runs of the same records
- [ ] Reader returns documents with identifiers and ordered passages
- [ ] Missing-FTS5 environment produces an actionable error (simulated in test)

**Verification:**
- [ ] Tests pass: `uv run pytest tests/artifact`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: `sqlite3` CLI on a written DB shows tables and passes
      `PRAGMA integrity_check`

**Dependencies:** Tasks 1, 3

**Files likely touched:**
- `src/papyrus_chat/artifact/records.py`
- `src/papyrus_chat/artifact/schema.py`
- `tests/artifact/test_schema.py`
- `tests/artifact/test_records.py`

**Estimated scope:** Medium

---

## Checkpoint B: After Tasks 3–4

- [ ] All four quality commands pass
- [ ] Artifact contract is creatable, readable, and validatable entirely in isolation
      (SPEC §4 independent testability)
- [ ] Proceed to Phase 3

---

## Phase 3: Builder core (first end-to-end path)

### Task 5: Safe XML parsing and DCLP adapter

**Description:** Implement `builder/xml.py` (hardened lxml parser factory: external
entity expansion and network resolution disabled per SPEC §6.3) and
`builder/collections/dclp.py`: parse each fixture/upstream DCLP record into
`DocumentRecord` + `IdentifierRecord` + `PassageRecord` — identifiers from all `idno`
types, title, metadata (inventory, material, origin place/date, keywords), declared
languages, edition passages with textpart structure and line/milestone references,
`display_text` retaining `supplied`/`unclear`/`gap`/`certainty` signals, normalized
`search_text` (display text never overwritten; unsupported structures warn rather than
invent, SPEC §6.3), and metadata-only records (empty `edition` div) kept discoverable.
Malformed XML raises a structured error carrying collection + path.

**Acceptance criteria:**
- [ ] Edition-text DCLP fixture extracts identifiers, metadata, language, and passages
      with textpart/line references
- [ ] Metadata-only fixture yields a discoverable document with zero passages
- [ ] `display_text` visibly retains supplied/unclear/gap/certainty signals; `search_text`
      is normalized and separate
- [ ] Malformed fixture produces a structured parse error naming collection and path
- [ ] A crafted fixture with an external entity / remote DTD reference is rejected safely
- [ ] Unsupported EpiDoc structures emit recorded warnings, never fabricated text

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder -k dclp`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: printed `display_text` for the supplied/unclear fixture reads naturally

**Dependencies:** Tasks 2, 4

**Files likely touched:**
- `src/papyrus_chat/builder/xml.py`
- `src/papyrus_chat/builder/collections/dclp.py`
- `tests/builder/test_dclp.py`

**Estimated scope:** Medium

---

### Task 6: Builder pipeline and CLI — first end-to-end artifact

**Description:** Implement `builder/pipeline.py` and the real `builder/cli.py`:
`papyrus-corpus-build dclp [--output PATH] [--source PATH] [--ref REF] [--force]`
builds an artifact from a local source directory (deliberate interim simplification —
plain directory; git-aware sources arrive in Task 10), reading files in lexicographic
path order (SPEC §6.2), writing manifest + `corpus.sqlite` + `ATTRIBUTION.md`
(CC BY 3.0, upstream link + README, contributing projects, commit, model-output
disclaimer per SPEC §7.4) into a temporary sibling directory moved into place
atomically, then validating before publication (SPEC §6.4). Print a completion report:
artifact path, collections, resolved source ref (commit recorded as the fixture
provenance commit for now), document/passage counts, size, elapsed time (SPEC §6.5);
logical hash placeholder is completed in Task 8. `--list-collections` from Task 1 keeps
working; unknown collections fail with supported names.

**Acceptance criteria:**
- [ ] `uv run papyrus-corpus-build dclp --source tests/fixtures/idp.data --output <tmp>`
      succeeds and produces exactly `manifest.json`, `corpus.sqlite`, `ATTRIBUTION.md`
- [ ] Manifest matches SPEC §7.2 structure with canonicalized collections
- [ ] Artifact passes `artifact.validation` (integrity + foreign keys)
- [ ] Completion report prints path, collections, counts, size, elapsed time
- [ ] Destination directory contains no source checkout, environment, key, or model cache
- [ ] Unknown collection argument fails with supported names and exit ≠ 0

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder`
- [ ] Build succeeds: `uv sync && uv run papyrus-corpus-build dclp --source tests/fixtures/idp.data --output /tmp/pc-check`
- [ ] Manual check: inspect printed report and `ATTRIBUTION.md` wording; `sqlite3
      /tmp/pc-check/corpus.sqlite "select count(*) from passages"` matches report

**Dependencies:** Tasks 3, 4, 5

**Files likely touched:**
- `src/papyrus_chat/builder/pipeline.py`
- `src/papyrus_chat/builder/cli.py`
- `tests/builder/test_pipeline.py`
- `tests/builder/test_cli_build.py`

**Estimated scope:** Medium

---

## Checkpoint C: After Tasks 5–6 — first end-to-end build (human review)

- [ ] All four quality commands pass
- [ ] Fixture directory → CLI → valid, searchable-by-SQLite artifact works end to end
- [ ] **Review with human before proceeding** (core architecture now established:
      records, schema, adapter protocol, pipeline shape)

---

### Task 7: Translations adapter and multi-collection builds

**Description:** Implement `builder/collections/translations.py` with the same adapter
protocol as DCLP: translation passages from `div type="translation"` (kind
`translation`, distinct from `edition`), textpart/line structure, declared languages,
identifiers across namespaces (apisid/controlNo/ddbdp/ddb-hybrid/HGV/TM), title/author
metadata. Register both adapters in a small collection registry so
`papyrus-corpus-build dclp translations` builds both collections into one artifact.
Per SPEC §3.1: collections are matched case-insensitively and canonicalized to
lowercase; the corpus is not presented as joined — links between records appear only
via explicit shared identifiers.

**Acceptance criteria:**
- [ ] Translation fixture extracts passages with kind `translation`, line/textpart
      references, and all identifier namespaces present in the file
- [ ] `DCLP`/`Translations`/mixed-case CLI arguments are accepted and canonicalized
- [ ] `dclp translations` build yields one artifact with both collections, editions and
      translations distinguishable in the passages table
- [ ] Documents sharing a TM/HGV identifier are linked in the identifiers table but not merged
- [ ] Build report lists both collections with per-collection counts

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder -k "translations or collections"`
- [ ] Build succeeds: `uv sync && uv run papyrus-corpus-build dclp translations --source tests/fixtures/idp.data --output /tmp/pc-check2`
- [ ] Manual check: mixed-case invocation (`DCLP Translations`) behaves identically

**Dependencies:** Tasks 5, 6

**Files likely touched:**
- `src/papyrus_chat/builder/collections/translations.py`
- `src/papyrus_chat/builder/collections/__init__.py` (registry)
- `src/papyrus_chat/builder/cli.py` (registry wiring)
- `tests/builder/test_translations.py`

**Estimated scope:** Medium

---

### Task 8: Determinism — logical content hash and reproducible builds

**Description:** Implement `artifact/hashing.py`: the canonical representation and
`sha256:<hex>` logical content hash over schema version, builder version, resolved
source commit, build-relevant options, and sorted documents/identifiers/passages —
excluding timestamps and SQLite page layout (SPEC §7.2, plan decision 6). Wire it into
the pipeline and manifest, and enforce sorted processing throughout (lexicographic file
order, canonical record ordering) so identical inputs and builder version always yield
the same hash. Update the completion report to include the hash.

**Acceptance criteria:**
- [ ] Canonicalization is documented in the module and pinned by golden-value tests
- [ ] Two builds from the same inputs produce identical logical content hashes
      (default-suite test, builds twice)
- [ ] Changing any document/passage/identifier, the source commit, or build options
      changes the hash; changing `created_at` does not
- [ ] Manifest and completion report both show the hash

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder -k determinism tests/artifact -k hashing`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: run the build twice into different outputs; compare printed hashes

**Dependencies:** Task 6 (Task 7 parallel-safe but reproducibility test should cover both
collections once Task 7 lands)

**Files likely touched:**
- `src/papyrus_chat/artifact/hashing.py`
- `src/papyrus_chat/builder/pipeline.py`
- `tests/artifact/test_hashing.py`
- `tests/builder/test_determinism.py`

**Estimated scope:** Medium

---

## Checkpoint D: After Tasks 7–8

- [ ] All four quality commands pass
- [ ] Both collections build; repeat builds are hash-reproducible
- [ ] Fixture-scale build + validation completes in seconds (early signal for SPEC §13)
- [ ] Proceed to Phase 4

---

## Phase 4: Builder robustness

### Task 9: Failure, replacement, and atomic assembly behavior

**Description:** Complete SPEC §6.4: if the output exists without `--force`, fail before
building anything; with `--force`, only the exact requested artifact directory is
replaced; the artifact is assembled and validated in a temporary sibling and moved
atomically, so a failure in acquisition/parsing/indexing/validation leaves any previous
artifact intact; a malformed source record fails the whole build (no partial artifact)
with its collection and path reported; no secrets are read or written during builds
(SPEC §6.4). CLI errors are concise and actionable, never tracebacks (SPEC §11).

**Acceptance criteria:**
- [ ] Existing output without `--force` fails fast with a message mentioning `--force`
- [ ] `--force` replaces only the named directory; sibling directories untouched
- [ ] Malformed fixture mid-corpus fails the build, names collection + path, and leaves
      any previous artifact at the output path fully intact and valid
- [ ] A forced rebuild interrupted before the atomic move leaves the old artifact intact
      (simulated in test)
- [ ] No LLM-related environment variables or credentials are read during a build
      (asserted in test)

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder -k "failure or force or atomic"`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: run a build onto an existing artifact without `--force`, then with;
      inspect behavior and messages

**Dependencies:** Task 6

**Files likely touched:**
- `src/papyrus_chat/builder/pipeline.py`
- `src/papyrus_chat/builder/cli.py`
- `tests/builder/test_failure.py`

**Estimated scope:** Medium

---

### Task 10: Local Git source resolution (commit-accurate reads)

**Description:** Implement `builder/source.py` for local sources per SPEC §6.2: the
`--source` path must be a Git checkout; `--ref` (default `master`) resolves to an exact
commit SHA (branches, tags, SHAs); files are read from the resolved commit's tree
(`git archive`/`ls-tree`/`cat-file` via subprocess — plan decision 3), so a dirty
working tree cannot change the build; the resolved SHA is recorded in the manifest
(replacing Task 6's interim placeholder); non-checkout paths and unresolvable refs fail
with actionable messages. Pipeline consumes the new source layer; fixture-directory
tests are updated to wrap fixtures in a temporary git repo.

**Acceptance criteria:**
- [ ] Local checkout source: `--ref` (branch, tag, short/full SHA) resolves and the full
      SHA lands in the manifest
- [ ] Uncommitted modifications to the working tree do not change the built content or
      logical hash (test edits a tracked fixture after committing)
- [ ] Non-git directory as `--source` fails with a clear message
- [ ] Unresolvable `--ref` fails with a clear message
- [ ] Builds remain hash-reproducible from the same commit

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder -k source`
- [ ] Build succeeds: `uv sync` (tests create throwaway git repos in `tmp_path`)
- [ ] Manual check: commit fixtures in a scratch repo, dirty the tree, rebuild —
      identical hash

**Dependencies:** Task 6

**Files likely touched:**
- `src/papyrus_chat/builder/source.py`
- `src/papyrus_chat/builder/pipeline.py`
- `tests/builder/test_source_local.py`

**Estimated scope:** Medium

---

### Task 11: Remote source acquisition (cache, partial clone, sparse checkout)

**Description:** Extend `builder/source.py` for remote `--source` URLs per SPEC §6.2:
use the `platformdirs` user cache directory, outside the artifact (plan decision 10); Git partial clone
(`--filter=blob:none`) with sparse-checkout limited to the selected collections so only
selected collection blobs are downloaded; fetch and resolve `--ref` to the full SHA;
reuse the cache across builds without modifying any user-owned checkout; the artifact
remains usable after the cache is deleted. Offline tests use a local bare repository as
the remote (see plan risk 1). Add one explicitly `network`-marked, default-excluded
smoke test against the real GitHub remote.

**Acceptance criteria:**
- [ ] Remote build (from test bare repo) fetches only selected-collection blobs
      (asserted via object counts or fetch size)
- [ ] `--ref` resolves to a full SHA recorded in the manifest
- [ ] Second build reuses the cache (no re-clone; refetch only)
- [ ] Deleting the cache does not affect an already-built artifact's usability
- [ ] No user checkout is ever modified
- [ ] `network`-marked smoke test exists, is excluded by default, and documents its
      manual invocation

**Verification:**
- [ ] Tests pass: `uv run pytest tests/builder -k remote`
- [ ] Build succeeds: `uv sync && uv run pytest -m network -k remote` (optional, real
      network; documents the GitHub path)
- [ ] Manual check: run the real remote build once (`--source
      https://github.com/papyri/idp.data.git --ref master`), note wall-clock time and
      cache behavior for the performance record

**Dependencies:** Task 10

**Files likely touched:**
- `src/papyrus_chat/builder/source.py`
- `tests/builder/test_source_remote.py`
- `tests/builder/test_network_smoke.py` (marker: `network`)

**Estimated scope:** Medium

---

## Checkpoint E: After Tasks 9–11 — full builder acceptance (human review)

- [ ] All four quality commands pass; default suite fully offline
- [ ] SPEC §15 "Builder" acceptance demonstrated from a clean clone:
      `uv sync`; `uv run papyrus-corpus-build dclp translations --output ./papyrus-corpus`
      succeeds without LLM configuration; only selected blobs fetched remotely; exact
      commit recorded and documented artifact files produced; repeat build → same
      logical hash; unknown collections / malformed records / existing outputs fail
      safely and clearly
- [ ] **Review with human before proceeding to retrieval**

---

## Phase 5: Retrieval

### Task 12: Identifier lookup

**Description:** Implement `retrieval/identifiers.py`: normalized exact identifier
lookup over the `identifiers` table (SPEC §8 step 1) — normalization rules (e.g.,
trim, casefold, collapse internal whitespace, strip zero-width characters; optionally
recognize `TM 23702`/`tm:23702`-style namespace prefixes) defined explicitly and
covered by unit tests; returns full documents (metadata-only included) with their
identifiers; deterministic ordering when several records match.

**Acceptance criteria:**
- [ ] Fixture-artifact lookups succeed for TM, ddbdp-style, and HGV identifiers in
      canonical and messy forms (case, spacing)
- [ ] Metadata-only documents are returned by identifier lookup
- [ ] Unknown identifiers return an empty result (not an error)
- [ ] Normalization rules are unit-tested against a documented list of input forms
- [ ] Lookup runs against the SQLite index (no table scan; `EXPLAIN QUERY PLAN` asserted
      in test)

**Verification:**
- [ ] Tests pass: `uv run pytest tests/retrieval -k identifiers`
- [ ] Build succeeds: `uv sync` (test fixture artifact built in-session from
      `tests/fixtures/idp.data`)
- [ ] Manual check: time identifier lookup on the fixture artifact — well under the
      100 ms SPEC §13 target

**Dependencies:** Task 6

**Files likely touched:**
- `src/papyrus_chat/retrieval/identifiers.py`
- `tests/retrieval/test_identifiers.py`

**Estimated scope:** Small

---

### Task 13: Full-text search and evidence packets

**Description:** Implement `retrieval/search.py` and `retrieval/evidence.py` per SPEC §8:
the search entry point attempts normalized identifier lookup first, then FTS5 search
over passage and document fields; user input is converted into a safe FTS5 query (no
query-syntax injection — quoted phrases/tokens); ranking uses an explicit fixed
BM25/FTS5 configuration with the `unicode61` tokenizer (plan decision 7), covered by
tests; diacritic-insensitive Greek matching comes from the shared normalization — both
`search_text` and query terms are case-folded and stripped of combining marks (NFD),
display text untouched; deterministic tie-breaking by
score, collection, document ID, passage sequence; corpus-wide search, per-document
scope, and collection/kind filters; metadata-only results when no passage text exists;
result snippets. `evidence.py` assembles the evidence packet: each item carries display
text, document metadata, passage kind, source locator (commit, path, textpart/line),
and a human-readable citation label.

**Acceptance criteria:**
- [ ] Identifier-shaped queries hit identifier lookup; free-text queries hit FTS5;
      both paths tested
- [ ] FTS5 query construction escapes/reserves FTS5 syntax; a query like `"drop OR NEAR("`
      is treated as literal text
- [ ] Ranking config (BM25 parameters, `unicode61` tokenizer) is explicit in code and pinned by tests
- [ ] Greek fixture text is findable with and without diacritics/case, via the shared
      NFD/case-fold normalization applied to both `search_text` and query terms
- [ ] Filters: collection, passage kind, and document scope all narrow results; tested
- [ ] Metadata-only documents surface for matching metadata searches
- [ ] Evidence items carry display text, kind, locator, and citation label; packet is a
      typed structure, not a text blob
- [ ] Deterministic ordering: same artifact + query → identical result sequence (test)

**Verification:**
- [ ] Tests pass: `uv run pytest tests/retrieval`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: sample queries against the fixture artifact; confirm snippet quality
      and citation labels; ordinary search under the 500 ms SPEC §13 target

**Dependencies:** Tasks 6, 12

**Files likely touched:**
- `src/papyrus_chat/retrieval/search.py`
- `src/papyrus_chat/retrieval/evidence.py`
- `tests/retrieval/test_search.py`
- `tests/retrieval/test_evidence.py`

**Estimated scope:** Medium

---

## Checkpoint F: After Tasks 12–13 — retrieval acceptance

- [ ] All four quality commands pass
- [ ] SPEC §15 "Artifact and retrieval" acceptance: artifact passes validation;
      searchable with source/cache removed; identifier and FTS search return stable,
      cited results; editions vs translations distinguishable; metadata-only records
      discoverable
- [ ] Proceed to Phase 6

---

## Phase 6: Chat runtime

### Task 14: LLM provider configuration and client

**Description:** Implement `chat/provider.py` per SPEC §9.2: configuration from
`LLM_BASE_URL` (required, trailing slash tolerated), `LLM_MODEL` (required),
`LLM_API_KEY` (optional — unauthenticated local servers supported); an `httpx` client
calling `chat/completions` relative to the API root; missing/unreachable/auth-failing/
rejecting endpoints mapped to concise, actionable user errors; the API key stays on the
server — never in logs, artifacts, HTML, or error pages; request/response validated at
the boundary. Tests use a real local mock HTTP server on a random port (stdlib
`http.server` in a test fixture thread — plan decision 9), never a paid endpoint.

**Acceptance criteria:**
- [ ] Missing `LLM_BASE_URL`/`LLM_MODEL` produce specific, actionable messages naming
      the variable
- [ ] Successful completion call returns message content (mock server happy path)
- [ ] Connection failure, HTTP 401/403, and 4xx/5xx provider rejections each map to
      distinct actionable messages without raw tracebacks
- [ ] API key is sent only in the `Authorization` header to the configured root; absent
      from logs in all tested paths
- [ ] Trailing-slash and no-slash base URLs behave identically

**Verification:**
- [ ] Tests pass: `uv run pytest tests/chat -k provider`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: run the app's provider check (from Task 16 onward) against a
      throwaway local server with/without env vars

**Dependencies:** Task 1

**Files likely touched:**
- `src/papyrus_chat/chat/provider.py`
- `tests/chat/test_provider.py`

**Estimated scope:** Medium

---

### Task 15: Evidence-grounded conversation

**Description:** Implement `chat/conversation.py` per SPEC §9.3: for each question,
retrieve a bounded evidence set locally (via Task 13); build a prompt containing only
the question, minimal conversation context, and the retrieved evidence with numbered
evidence markers; a system instruction that (a) distinguishes corpus statements from
inference, (b) requires inline markers mapping to evidence items, (c) frames corpus
text and model output as untrusted data, never instructions (SPEC §9.3 last paragraph);
parse the answer's markers back to evidence items; when evidence is insufficient, the
answer says so and suggests a narrower search — and must never invent a papyrus text
for metadata-only records; published translations vs model-generated prose are
distinguished in the returned structure. All tested against the mock provider.

**Acceptance criteria:**
- [ ] Prompt contains question + bounded evidence + marker scheme; no full-corpus dump
- [ ] Answer markers map back to exactly the evidence items supplied (test with mock
      responses including multiple markers and a no-marker response)
- [ ] Insufficient-evidence path returns a "not enough evidence" style answer with a
      narrowing suggestion; metadata-only records are never given invented text
- [ ] A malicious corpus fixture containing instruction-like text ("ignore previous
      instructions…") is passed through as inert evidence; model output is never
      executed or treated as commands
- [ ] The returned structure separates source-translation content from
      model-generated content with explicit labels
- [ ] Conversation context is the client-supplied rolling history (plan decision 11),
      trimmed server-side to a bounded size and included only as needed for coherence

**Verification:**
- [ ] Tests pass: `uv run pytest tests/chat -k conversation`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: headless script or test logs showing the exact prompt and evidence
      set for a sample question

**Dependencies:** Tasks 13, 14

**Files likely touched:**
- `src/papyrus_chat/chat/conversation.py`
- `tests/chat/test_conversation.py`

**Estimated scope:** Medium

---

## Checkpoint G: After Tasks 14–15 — grounded chat (headless)

- [ ] All four quality commands pass
- [ ] Question → retrieval → evidence-marked prompt → cited answer works against the
      mock provider, entirely offline
- [ ] Marker→citation mapping and labeling proven by tests
- [ ] Proceed to Phase 7

---

## Phase 7: Web UI

### Task 16: Web application skeleton and papyrus-chat CLI

**Description:** Implement the real `chat/cli.py` and `web/application.py` per SPEC §9.1
and §10: `papyrus-chat --artifact PATH [--host 127.0.0.1] [--port 8000] [--no-open]`
validates manifest, schema compatibility, required files, SQLite integrity, and
provider configuration **before** opening the browser; serves the FastAPI app via
Uvicorn bound to `127.0.0.1` by default; opens the default browser unless `--no-open`;
base template with semantic HTML, one local stylesheet, Jinja2 autoescaping on, and
plain-language terminology ("Search the corpus", "Evidence used" — never
"chunks"/"vector"/"context window", SPEC §10). Startup validation failures exit with
concise actionable messages.

**Acceptance criteria:**
- [ ] Valid artifact + provider env: app starts, browser opens (unless `--no-open`),
      index route renders with sensible empty state
- [ ] Incompatible artifact schema major → actionable rejection message (SPEC §7.2)
- [ ] Missing/corrupt artifact files, failed integrity check, and missing provider
      config each fail before the browser opens, with distinct actionable messages
- [ ] Server binds to `127.0.0.1` by default; `--host`/`--port` honored
- [ ] Rendered HTML passes basic semantic checks (lang attr, title, form labels) in tests
- [ ] Templates autoescape: a fixture title containing `<script>` renders inert

**Verification:**
- [ ] Tests pass: `uv run pytest tests/web`
- [ ] Build succeeds: `uv sync && uv run papyrus-chat --artifact <fixture artifact> --no-open`
      (background, then request `/`)
- [ ] Manual check: startup with a broken artifact dir and with missing env vars —
      messages are helpful, no tracebacks

**Dependencies:** Tasks 3, 6, 14

**Files likely touched:**
- `src/papyrus_chat/chat/cli.py`
- `src/papyrus_chat/web/application.py`
- `src/papyrus_chat/web/templates/base.html`, `index.html`
- `tests/web/test_application.py`

**Estimated scope:** Medium

---

### Task 17: Search interface

**Description:** Implement the search page per SPEC §10: prominent search field;
collection and passage-kind filters; results list showing identifier, title, collection,
passage kind, and a short escaped snippet; result links to the document view (Task 18
route stubbed if needed); plain-language guidance. Search must work with no LLM
configured or contacted (SPEC §9, §15). Identifier-shaped input hits identifier lookup;
everything else hits FTS (Task 13 ordering).

**Acceptance criteria:**
- [ ] Prominent search field with associated label; filters are real form controls
- [ ] Results show identifier, title, collection, kind, snippet; every result links to
      its document
- [ ] Empty/unknown queries render a friendly no-results state, no traceback
- [ ] Corpus and snippet content is escaped in HTML (test with `<script>` fixture text)
- [ ] A search request completes with provider env vars unset — no LLM contact
      (asserted: no provider call made)
- [ ] Filter combinations (collection, kind) behave as in Task 13 tests

**Verification:**
- [ ] Tests pass: `uv run pytest tests/web -k search`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: in the browser, search a Greek fixture phrase and an identifier;
      apply filters; keyboard-navigate the results (Tab/Enter)

**Dependencies:** Tasks 13, 16

**Files likely touched:**
- `src/papyrus_chat/web/application.py` (search route)
- `src/papyrus_chat/web/templates/search.html`
- `src/papyrus_chat/web/static/style.css`
- `tests/web/test_search.py`

**Estimated scope:** Medium

---

## Checkpoint H1: After Tasks 16–17 — search in the browser, no LLM

- [ ] All four quality commands pass
- [ ] Browser journey: start app → search by text and by identifier → filter → open a
      result — all without any LLM configuration
- [ ] First search results appear with no LLM call (SPEC §13)
- [ ] Proceed within Phase 7

---

### Task 18: Document view

**Description:** Implement the selected-document view per SPEC §10: metadata (title,
identifiers with namespaces, origin, material, keywords, languages), all passages with
kind labels distinguishing **edition** vs **source translation**, visible **metadata
only** label for textless DCLP records (SPEC §3.1), provenance display (collection,
source identifier, exact commit, source-relative path, textpart/line locator per
SPEC §7.4), and copyable citations (identifier + locator). Edition and translation
passages render their display text with uncertainty signals intact; polytonic Greek is
legible via the bundled polytonic-Greek-capable webfont (e.g., Noto Sans with its OFL
license file; plan decision 8), with system-font fallback for papyrological symbols
the bundled font lacks.

**Acceptance criteria:**
- [ ] Document route renders metadata + passages for a fixture document with edition text
- [ ] Metadata-only fixture document renders with a visible "metadata only" label and no
      passage section errors
- [ ] Translation passages are labeled as source translations; uncertainty markup
      signals remain visible in display text
- [ ] Provenance block shows commit, path, and locator; a copyable citation string is
      present (button/element with the exact text)
- [ ] All dynamic content escaped; identifier namespaces displayed per record
- [ ] Deep link to a document works from a search result and stays stable on reload
- [ ] Bundled webfont (with its license file) is served locally; nothing is fetched
      from external font sources

**Verification:**
- [ ] Tests pass: `uv run pytest tests/web -k document`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: open the metadata-only fixture doc and a translation doc in the
      browser; copy a citation; verify Greek renders legibly

**Dependencies:** Tasks 13, 16

**Files likely touched:**
- `src/papyrus_chat/web/application.py` (document route)
- `src/papyrus_chat/web/templates/document.html`
- `src/papyrus_chat/web/static/fonts/**` (bundled font + license)
- `tests/web/test_document.py`

**Estimated scope:** Medium

---

### Task 19: Chat panel and Evidence used

**Description:** Implement the chat panel per SPEC §10 and SPEC §9.3: chat operates
over the selected document or the whole corpus (explicit scope selector); each answer
renders with an expandable **Evidence used** section showing the exact evidence items
supplied to the model, with markers in the answer linked/annotated to their evidence
items; **source translation** and **model-generated** labels appear where applicable;
insufficient-evidence answers say so and suggest a narrower search; provider failures
(config missing, unreachable, auth, rejection) render as actionable inline messages;
chat history is the decided client-side rolling history sent with each question (plan
decision 11). Small vanilla JS only (form
submission, evidence toggle via `<details>` needs none).

**Acceptance criteria:**
- [ ] Asking a corpus-scoped question returns an answer with visible evidence markers
      mapped to evidence items (integration with Task 15, via mock provider in tests)
- [ ] "Evidence used" expands to the exact evidence sent to the model, with citation
      labels and locators
- [ ] Model-generated prose is visibly labeled; published translations shown as source
      translations
- [ ] Insufficient-evidence answer displays the suggested narrowing, never invented text
- [ ] Each provider failure mode renders a distinct actionable message in the page
- [ ] Chat over a selected document restricts evidence to that document (asserted)
- [ ] No page content or request leaks the API key (asserted across responses)

**Verification:**
- [ ] Tests pass: `uv run pytest tests/web -k chat`
- [ ] Build succeeds: `uv sync && uv run papyrus-chat --artifact <fixture artifact> --no-open`
      with mock/real local provider
- [ ] Manual check: full browser conversation against a local OpenAI-compatible server;
      expand evidence; try a question with no matching evidence

**Dependencies:** Tasks 15, 18

**Files likely touched:**
- `src/papyrus_chat/web/application.py` (chat routes)
- `src/papyrus_chat/web/templates/_chat.html`, `index.html`/`document.html` wiring
- `src/papyrus_chat/web/static/app.js`
- `tests/web/test_chat.py`

**Estimated scope:** Medium

---

### Task 20: Accessibility, escaping, and security audit pass

**Description:** Cross-cutting audit of the finished UI against SPEC §10's interface
requirements: full keyboard navigation (every interactive control reachable and
operable); semantic HTML with associated labels everywhere; sufficient contrast without
color-alone signals; polytonic Greek and common papyrological symbols legible; usable
on laptop and narrow windows; all corpus/model output escaped; `127.0.0.1`-only
default; zero third-party/telemetry requests (assert no external URLs in rendered
pages or static assets); API key absent from all responses and logs. Fixes land as
small edits across templates/routes plus regression tests.

**Acceptance criteria:**
- [ ] Keyboard-only walkthrough of search → results → document → chat → evidence
      documented and succeeds (manual), with focus states visible
- [ ] Automated checks: every form control has an associated label; heading hierarchy
      valid; `lang` attributes correct for Greek passages where applicable
- [ ] Automated checks: response bodies never contain the API key value; no external
      resource references in any rendered page or asset
- [ ] Contrast spot-checks pass without relying on color alone (e.g., labels differ by
      more than color)
- [ ] Narrow-window (≤ 480 px) layout remains usable (manual check documented)
- [ ] Regression tests added for every escaping/labeling defect found and fixed

**Verification:**
- [ ] Tests pass: `uv run pytest tests/web`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: keyboard walkthrough + narrow window, findings recorded (fixed or
      filed as spec deviations)

**Dependencies:** Tasks 17, 18, 19

**Files likely touched:**
- `src/papyrus_chat/web/templates/**` and `static/**` (small fixes)
- `tests/web/test_accessibility.py`
- `tests/web/test_security.py`

**Estimated scope:** Small

---

## Checkpoint H2: After Tasks 18–20 — full UI

- [ ] All four quality commands pass
- [ ] SPEC §15 "Chat and interface" acceptance demonstrated in the browser: startup with
      the three env vars; search without LLM contact; cited answers with exact evidence;
      useful messages for missing evidence/config/auth/incompatibility; key absent from
      responses/logs/artifact; local-only default server
- [ ] Proceed to Phase 8

---

## Phase 8: Release readiness

### Task 21: End-to-end integration test

**Description:** Add the SPEC §12.2 integration test: in one offline test session,
initialize a git repo from `tests/fixtures/idp.data` (per plan decision 4), run the
real `papyrus-corpus-build` CLI (both collections), validate the artifact, start the
application via its ASGI/TestClient surface with the mock LLM server from Task 14/15,
then exercise search, document view, and a chat answer — asserting citations map to
displayed evidence end to end. Also asserts the artifact remains searchable after the
source repo and cache directories are deleted (SPEC §15).

**Acceptance criteria:**
- [ ] One test (or tightly coupled set) covers build → validate → serve → search →
      document → chat with mock provider, all offline
- [ ] Artifact stays searchable after source checkout and cache removal
- [ ] The chat answer's displayed citations correspond to the exact evidence items sent
      (asserted from the mock server's recorded request)
- [ ] Test runs in the default suite within a reasonable time budget

**Verification:**
- [ ] Tests pass: `uv run pytest tests/integration`
- [ ] Build succeeds: `uv sync`
- [ ] Manual check: `uv run pytest` (whole default suite) green and offline

**Dependencies:** Tasks 10, 19

**Files likely touched:**
- `tests/integration/test_end_to_end.py`
- `tests/conftest.py` (shared session fixtures if needed)

**Estimated scope:** Medium

---

### Task 22: README, performance record, and release sweep

**Description:** Write `README.md` reproducing the two-command journey from a clean
clone (prerequisites, `uv sync`, both commands with the three env vars) and linking to
`SPEC.md` (SPEC §15 Quality). Record measured performance on the local development
machine as the documented reference machine (plan resolved decision 4): warm-cache
fixture and real-corpus build times (network
time reported separately), app startup + validation time, identifier lookup and typical
FTS search latency, first-progress latency (SPEC §13; misses documented with
bottlenecks before any target change). Final sweep: all SPEC §15 acceptance criteria
checked off with evidence, all four quality commands green, default suite offline,
`network` tests still excluded by default.

**Acceptance criteria:**
- [ ] README documents the complete clean-clone journey and links to `SPEC.md` and
      `docs/performance.md`
- [ ] Performance measurements recorded for the local development machine (the named
      reference machine), including network-separated remote build time
- [ ] Any missed SPEC §13 target documented with measured result and bottleneck
- [ ] SPEC §15 checklist reviewed item by item; every item demonstrated or explicitly
      tracked as a deviation
- [ ] `uv run pytest && uv run ruff check . && uv run ruff format --check . &&
      uv run ty check` all green; default suite offline

**Verification:**
- [ ] Tests pass: `uv run pytest` (full suite)
- [ ] Build succeeds: the README's own commands executed from a clean clone
- [ ] Manual check: follow the README exactly in a fresh checkout; confirm the
      two-command journey works

**Dependencies:** Task 21

**Files likely touched:**
- `README.md`
- `docs/performance.md`
- `tasks/plan.md` (status + resolved open questions)

**Estimated scope:** Small

---

## Checkpoint I: After Tasks 21–22 — release readiness (human review)

- [ ] All four quality commands pass; full default suite offline and green
- [ ] All SPEC §15 acceptance criteria demonstrated with evidence
- [ ] Performance recorded on a documented reference machine
- [ ] README reproduces the journey from a clean clone
- [ ] **Final review with human before tagging a proof-of-concept release**

---

## Status legend

- `[ ]` pending · `[x]` done · `[~]` in progress (update in place)
- Checkpoints with **human review** block further work until approved
