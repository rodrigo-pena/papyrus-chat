# Implementation Plan: Papyrus Chat proof of concept

Status: Decisions recorded 2026-08-31 — open questions resolved, ready for implementation
Spec: `SPEC.md` (Accepted, 2026-08-31)
Task list: `tasks/todo.md`

## Overview

Build the Papyrus Chat proof of concept from a clean clone: a deterministic corpus builder
(`papyrus-corpus-build`) that turns selected `idp.data` collections (`dclp`, `translations`)
into a self-contained, versioned SQLite+FTS5 artifact, and a local web application
(`papyrus-chat`) that searches that artifact and answers questions through an
OpenAI-compatible endpoint with visible, cited evidence. The repo is currently greenfield:
only `SPEC.md`, `README.md`, and `.gitignore` exist.

## Working agreements

- Quality gate for every task (SPEC §12.3): `uv run pytest`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run ty check` — all four must pass before a task is
  done. Focused test commands are listed per task in `tasks/todo.md`.
- Default test suite is offline: no live `idp.data` clone, no external LLM calls
  (SPEC §12.2). Network tests carry the `network` pytest marker and are excluded by default.
- Commits: Conventional Commit messages in semantically sensible chunks, only when the
  user explicitly allows committing.
- Spec is the contract (SPEC §17): any change to scope, commands, artifact compatibility,
  failure behavior, security boundaries, or acceptance criteria updates `SPEC.md` in the
  same change and is called out in review.
- Update `tasks/todo.md` checkboxes as tasks complete; keep this plan updated with
  decisions discovered during implementation.

## Architecture decisions

1. **One distribution, two console entry points** (SPEC §5.2): `pyproject.toml` with
   `papyrus-corpus-build = "papyrus_chat.builder.cli:app"` and
   `papyrus-chat = "papyrus_chat.chat.cli:app"`. Python ≥ 3.12; versions locked in `uv.lock`.
2. **Frozen Pydantic models at capability boundaries** (SPEC §11). Minor layout addition
   beyond the SPEC §5.2 tree (which is a SHOULD): `artifact/records.py` for the
   `DocumentRecord` / `PassageRecord` / `IdentifierRecord` boundary models, and
   `builder/xml.py` for a hardened `lxml` parser factory shared by adapters. Both follow
   the spec's own "small cohesive modules" rule.
3. **Git via `subprocess`, not pygit2.** Git is already a builder prerequisite for remote
   clones; subprocess avoids a native dependency. Local checkouts are read
   commit-accurately (`git archive`/`ls-tree`/`cat-file` of the resolved SHA), never from
   a dirty working tree (SPEC §6.2).
4. **Deliberate interim simplification:** Task 6's first end-to-end pipeline reads a plain
   fixture directory as its source. Task 10 replaces this with git-aware local sources
   (checkout required, commit-accurate reads), and Task 21's integration test uses a
   git-initialized fixture repo. This keeps the first vertical slice small without ever
   shipping the simplification.
5. **Stable IDs** derive from collection + source-relative path (+ structural locator for
   passages), never insertion order (SPEC §7.3). Source paths are unique per collection
   (one EpiDoc file = one document), so the path is the natural stable key.
6. **Logical content hash:** canonical JSON (sorted keys, UTF-8, compact separators, no
   timestamps, no SQLite layout) over schema version, builder version, resolved source
   commit, build-relevant options, and sorted documents/identifiers/passages →
   `sha256:<hex>` (SPEC §7.2). Algorithm documented in `artifact/hashing.py`.
7. **FTS5 tokenizer: `unicode61`** (decided 2026-08-31), with an explicit, tested BM25
   configuration (SPEC §8). unicode61 case-folds but does not strip diacritics, so
   diacritic-insensitive Greek search comes from shared normalization: both
   `search_text` and user query terms are case-folded and stripped of combining marks
   (NFD) before indexing and searching. Display text is never modified (SPEC §6.3);
   substring search is out of scope for the POC.
8. **Web UI:** server-rendered Jinja2 with autoescape on, semantic HTML,
   `<details>/<summary>` for evidence expansion (keyboard-friendly without JS), one local
   CSS file, and a small vanilla JS file. No Node, no CDN, no third-party requests
   (SPEC §5.1, §10). A polytonic-Greek-capable webfont (e.g., Noto Sans) is bundled in
   `web/static/fonts/` with its OFL license file (decided 2026-08-31); system fonts
   remain the fallback for papyrological symbols the bundled font lacks.
9. **Mock LLM provider in tests:** a real local HTTP server on a random port using the
   stdlib `http.server` in a test fixture thread — satisfies "mock HTTP server" (SPEC
   §12.2) without a new dependency and without touching a paid endpoint.
10. **User cache location** for remote clones (SPEC §6.2): the `platformdirs` runtime
    dependency (decided 2026-08-31) provides the platform-appropriate cache directory.
11. **Conversation state lives in the browser:** the server is stateless; a short rolling
    history of questions/answers is kept client-side and sent with each question
    (decided 2026-08-31). The server trims it to the bounded context coherence requires.

## Dependency graph

Built bottom-up per the capability map (SPEC §4); implementation order follows the arrows.

```
Project skeleton + tooling (T1)          EpiDoc fixtures (T2)
        │                                       │
        └───────────────┬───────────────────────┘
                        │
              artifact-format (T3 manifest+validation, T4 schema+records)
                        │
        ┌───────────────┼────────────────────────────┐
        │               │                            │
   DCLP adapter (T5)    │                     (parallel branch)
        │               │                    provider client (T14)
   pipeline+CLI (T6) ───┤                            │
        │               │                            │
   Translations (T7)    │                            │
   Determinism (T8)     │                            │
   Failure/atomicity (T9)│                           │
   Local git source (T10)│                           │
   Remote source (T11)   │                           │
        │               │                            │
        └──────► retrieval (T12 identifiers, T13 FTS+evidence)
                        │                            │
                        └────────► grounded conversation (T15) ◄─┘
                                       │
                       web app skeleton + chat CLI (T16)
                                       │
                       search page (T17) → document view (T18) → chat panel (T19)
                                       │
                       a11y/security audit (T20)
                                       │
                       integration test (T21) → README + performance + sweep (T22)
```

## Vertical slicing strategy

The spec's capability map is layered, but the plan slices user-visible increments:

- **Phase 3 delivers the first complete user path** (fixture directory → `papyrus-corpus-build`
  → valid artifact) before any robustness work. Tasks 3–4 are the shared foundation the
  dependency graph genuinely requires first (both builder and retrieval consume the
  artifact contract), and each is independently testable as SPEC §4 demands.
- Each subsequent task completes a behavior the spec names (second collection,
  reproducibility, failure safety, real git sources, identifier lookup, search, chat,
  each UI surface) rather than finishing a horizontal layer.
- Every task leaves the repo green: all four quality commands pass at each task boundary.

## Task list (index)

Tasks with full acceptance criteria, verification, dependencies, and file lists live in
`tasks/todo.md`.

| Phase | Tasks | User-visible increment |
|-------|-------|------------------------|
| 1. Foundation | 1–2 | `uv run` works; both CLIs respond; representative fixtures committed |
| 2. Artifact format | 3–4 | Artifact contract creatable and validatable in isolation |
| 3. Builder core | 5–8 | End-to-end build of both collections from fixtures; reproducible hash |
| 4. Builder robustness | 9–11 | Full SPEC §15 "Builder" acceptance, remote GitHub source |
| 5. Retrieval | 12–13 | SPEC §15 "Artifact and retrieval" acceptance |
| 6. Chat runtime | 14–15 | Grounded, cited answers against a mock provider (headless) |
| 7. Web UI | 16–20 | Full browser interface; search without LLM; a11y/security audit |
| 8. Release | 21–22 | Offline integration test; README; performance record; final sweep |

Checkpoints after every 2–3 tasks (see `tasks/todo.md`): A (foundation), B (artifact
format), C (**first end-to-end build — human review**), D (both collections + determinism),
E (**full builder acceptance — human review**), F (retrieval acceptance), G (grounded chat),
H1 (search in browser), H2 (full UI + audit), I (**release readiness — human review**).

## Parallelization opportunities

With multiple agents/sessions (respecting task dependencies):

- T2 (fixtures) ∥ T1 (skeleton).
- T7 (Translations adapter) ∥ T8/T9/T10 — independent once the T5 adapter protocol and
  T6 registry exist.
- T12/T13 (retrieval) ∥ T9/T10/T11 — retrieval only needs artifacts from T6/T7.
- T14 (provider) ∥ T12/T13 — no shared state with retrieval.
- T17 (search page) ∥ T18 (document view) after T16 + T13, coordinating route and
  template conventions first.
- T20 (audit fixes) partially ∥ T21 (integration test).

Must stay sequential: T3→T4→T5→T6 (core chain), T10→T11 (source layer), T14→T15,
T16→T17→T19 (UI chain), T21→T22.

## Upstream data reference (verified 2026-08-31)

Findings from the live `papyri/idp.data` repository (default branch `master`), used by the
fixture and adapter tasks:

- **DCLP layout:** `DCLP/<numeric-prefix>/<tm-number>.xml` (e.g., `DCLP/23/23702.xml`),
  sharded into numeric prefix directories.
- **DCLP identifiers observed:** `<idno type="dclp">`, `TM`, `LDAB`, `dclp-hybrid`,
  `filename`; metadata in `msDesc` (inventory no, material, origin place/date with
  `notBefore`/`notAfter`, provenance, keyword terms).
- **Metadata-only DCLP records are real:** `DCLP/23/23702.xml` has
  `<div type="edition" xml:space="preserve"/>` — an empty edition div. The adapter must
  treat this as a metadata-only document, not an error (SPEC §3.1).
- **Translations layout:** `Translations/<numeric-prefix>/<ddbdp-id>-<sequence>.xml`
  (e.g., `Translations/3/3643-1.xml`, also letter-suffixed ids like `3662a-1.xml`).
  Multiple files can translate the same text (sequence `-1`, `-2`); each file is one
  document.
- **Translations identifiers observed:** `apisid`, `controlNo`, `ddbdp` (e.g.,
  `p.tebt.1.7`), `ddb-perseus-style`, `ddb-hybrid`, `HGV`, `TM`, `filename`; passages in
  `<div type="translation" xml:lang="en"><ab>…</ab></div>`; declared language in
  `langUsage`.
- **License statement inside records:** CC BY 3.0 (DCLP/LDAB/Trismegistos; APIS for
  translations) — consistent with SPEC §7.4 attribution requirements.
- DCLP records with actual edition text use `<div type="edition">` with `textpart`
  divs, `lb` milestones, and `supplied`/`unclear`/`gap`/`certainty` markup; fixture
  selection in T2 must include at least one such record plus the categories in SPEC §12.1.

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Partial-clone (`--filter=blob:none`) may not behave over local `file://` remotes in offline tests | Med | Verify git ≥ 2.19 supports filters over file:// early in T11; fallback: local bare repo as remote for acquisition tests, assert sparse-checkout + blob counts separately; keep a `network`-marked smoke test for the real GitHub remote |
| unicode61 does not fold Greek diacritics and offers no substring search | Med | Diacritic-insensitive matching via shared NFD/case-fold normalization of `search_text` and query terms (plan decision 7); explicit, tested config in T13; substring search explicitly out of POC scope |
| FTS5 unavailable in the runtime `sqlite3` module | Low-Med | uv-managed CPython builds enable FTS5; add a startup capability check with an actionable message (T4), fail fast |
| LLM ignores evidence-marker instructions; output format drifts across providers | Med | Tolerant marker parsing, explicit prompt contract, "insufficient evidence" fallback, all tested against the mock server (T15); no reliance on provider-specific SDK behavior (SPEC §9.2) |
| Prompt injection via corpus text or model output | Med | Untrusted-content framing in system prompt, evidence sent as delimited data, no tool execution exists by design; malicious fixture + test in T15/T20 (SPEC §9.3) |
| Logical hash not reproducible across machines/runs | Med | Canonical form pinned in T8 (sorted, UTF-8, compact JSON, excludes timestamps/SQLite layout); build-twice test in the default suite; record any cross-platform deviation |
| §13 performance targets unmeasured until late | Med | Measure at fixture scale at checkpoints C and D; full-corpus measurements recorded in T22; per SPEC §13, misses are documented before targets change |
| Fixture licensing/provenance handled incorrectly | Low | CC BY 3.0 attribution + upstream commit/paths recorded in `tests/fixtures/idp.data/PROVENANCE.md` (T2, SPEC §12.1); mirrors SPEC §7.4 wording |
| Translations multi-file identity (same text, `-1`/`-2` files) mishandled | Low | Stable ID = collection + source path (one file = one document); cross-record links only via explicit shared identifiers (SPEC §3.1) |
| Remote build exceeds the 15-minute SHOULD target | Low | Partial clone + sparse checkout mandated (T11); network time reported separately (SPEC §13); measured and documented in T22 |

## Resolved decisions (recorded 2026-08-31 after human review)

1. **FTS5 tokenizer: `unicode61`** — diacritic-insensitive Greek search via shared
   `search_text`/query normalization (decision 7).
2. **Webfont: bundled** — a polytonic-Greek-capable font (e.g., Noto Sans, OFL) ships
   in `web/static/fonts/` with its license; system fallback for uncovered symbols
   (decision 8).
3. **Cache location: `platformdirs`** added as a runtime dependency (decision 10).
4. **Reference machine: the local development machine** — measurements recorded in
   `docs/performance.md`, linked from the README (Task 22).
5. **Fixtures: pinned with an override path** — the fixture set records a pinned
   upstream commit SHA and provenance; a documented refresh mechanism lets an informed
   user regenerate fixtures from HEAD of the upstream default branch (`master`) or an
   arbitrary commit SHA, updating `PROVENANCE.md` (Task 2; network-dependent, never
   part of the default suite).
6. **Chat history: short rolling history kept client-side**, sent with each question;
   the server stays stateless (decision 11).

## Definition of done for the plan

- [x] Every task has acceptance criteria and verification steps (`tasks/todo.md`)
- [x] Dependencies identified and ordered; no task exceeds ~5 meaningful files
      (T1/T2 are bootstrap/data exceptions, flagged in their entries)
- [x] Checkpoints between phases, human review at C, E, and I
- [x] Human reviewed the plan and resolved all open questions (2026-08-31);
      implementation may begin at Task 1
