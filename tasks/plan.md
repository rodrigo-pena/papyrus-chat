# Implementation Plan: Local STDIO MCP Server for Papyrus Chat

## Overview

Expose one validated local corpus artifact through a deterministic, read-only
`papyrus-mcp` STDIO server. The host agent remains responsible for natural-
language interpretation and all LLM or web calls; the server performs local
retrieval only.

## Architecture Decisions

- Extract transport-neutral corpus models, projections, and lifecycle management
  into `papyrus_chat.corpus`; retain legacy retrieval and Pydantic-AI imports as
  compatibility re-exports.
- Bind `CorpusService.open()` to one resolved artifact root and open its SQLite
  database read-only with serialized access to the connection and lazy semantic
  encoder.
- Keep MCP imports lazy so the base installation remains usable and
  `papyrus-mcp --help` does not require the MCP extra.
- Expose only bounded, structured, read-only tools over STDIO. No HTTP, REST,
  authentication, plugins, downloading, or artifact version management.

## Task List

### Phase 1: Foundation

- [ ] Extract `CorpusService`, shared result models, and projections.
- [ ] Preserve chat/agent behavior through compatibility aliases.

### Checkpoint: Foundation

- [ ] Existing retrieval, agent, and web tests pass.
- [ ] Read-only access and service cleanup are covered.

### Phase 2: Discovery Contracts

- [ ] Add corpus provenance/capability information.
- [ ] Add bounded identifier lookup and SQL-bounded facets.
- [ ] Add explicit semantic suggestion availability states.
- [ ] Enforce the minimum inspection ID bound.

### Checkpoint: Core Contracts

- [ ] Focused corpus and retrieval tests pass.
- [ ] Existing Pydantic-AI tool projections remain unchanged.

### Phase 3: MCP Server

- [ ] Add optional MCP dependency and `papyrus-mcp` entry point.
- [ ] Register six annotated structured-output tools over STDIO.
- [ ] Test in-memory and subprocess MCP clients, including errors and no-network operation.

### Phase 4: User Guidance

- [ ] Add the instruction-only `research-papyri` skill.
- [ ] Add and link the local MCP integration guide.

### Checkpoint: Complete

- [ ] Full test, lint, format, and type checks pass.
- [ ] Five conventional commits are present locally.
- [ ] No remote push is performed.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| MCP SDK API differs across v2 releases | High | Pin `mcp>=2,<3`, verify against the installed SDK, and contract-test schemas. |
| Legacy tests mutate direct retrieval connections | Medium | Keep the legacy constructor compatible while runtime `CorpusService.open()` uses URI read-only mode. |
| Lazy semantic model loading can race with concurrent calls | Medium | Serialize service calls and semantic initialization with one re-entrant lock. |
| STDIO diagnostics corrupt protocol frames | High | Route diagnostics to stderr and test a real subprocess client. |

## Open Questions

- None blocking implementation; the pasted feature brief supplies the required public choices.
