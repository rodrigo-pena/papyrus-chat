# Performance measurements

These measurements were recorded on 2026-08-31 on an Apple M5 Max with 128
GB RAM, macOS, Python 3.13, and a broadband connection. The full-corpus
baseline below predates the schema-v2 DDbDP/HGV implementation and is retained
as a reference; it is not presented as a fresh v2 benchmark.

## Full-corpus baseline

The measured baseline built the real `dclp` collection with the v0.1.0
builder. It covered 14,842 documents and produced a 73.4 MB schema-v1
artifact.

| Measurement | Result | Target | Status |
|-------------|--------|--------|--------|
| Remote `dclp` build, cold cache | 565–608 s across two runs | ≤ 15 min | pass |
| Remote `dclp` rebuild, warm cache | 545 s | ≤ 5 min | miss |
| Artifact startup and validation | 864 ms | ≤ 5 s | pass |
| Identifier lookup (`TM 23944`, average of 50) | 0.027 ms | ≤ 100 ms | pass |
| FTS search, polytonic Greek `ἔτους` (average of 50) | 16.4 ms | ≤ 500 ms | pass |
| FTS search, English `horoscope` (average of 50) | 14.9 ms | ≤ 500 ms | pass |
| First search without an LLM call | Search never contacts the LLM | required | pass |

The warm rebuild bottleneck was one `git show` subprocess per file in
`LocalGitSource.read_bytes` (about 14,800 process spawns). Batch reads through
`git archive` or a persistent `git cat-file` process remain a future
optimization. DDbDP plus HGV and the v2 full-corpus rebuild were not measured
end-to-end in this implementation session.

## v2 validation

The committed automated coverage builds the paired real-source DDbDP/HGV
fixture, validates schema-v2 tables and links, and exercises distinct-document
FTS, date overlap, language filters, facets, and bounded evidence. A full
68,000-document DDbDP/67,000-record HGV benchmark remains an explicit follow-up
when a pinned upstream checkout and measurement window are available.

## Acceptance feedback

No live collaborator conversation or feedback session was available during
this implementation run. The four-query walkthrough is documented in the
README and covered at the contract level by deterministic `FunctionModel`
dialogue evaluations; it still needs a human smoke run against a configured
tool-calling provider.
