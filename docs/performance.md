# Performance measurements

Reference machine (SPEC 13): local development machine — Apple M5 Max,
128 GB RAM, macOS, Python 3.13 (uv-managed), broadband connection.
Measured: 2026-08-31, papyrus-chat 0.1.0.

## Corpus builds

| Measurement | Result | SPEC 13 target | Status |
|-------------|--------|----------------|--------|
| Real remote build, `dclp` only (cold cache; partial clone + sparse checkout + parse + index) | 565–608 s across two runs | first remote build ≤ 15 min (SHOULD) | pass |
| Real remote rebuild, `dclp` only (warm cache: no re-clone; parse + index again) | 545 s | warm build ≤ 5 min | **miss** |
| Fixture-scale build (4 documents, 2 collections, warm) | ~0.1 s | — | — |
| Rebuild determinism at corpus scale (14,842 documents) | identical logical content hash on rebuild | identical hash required | pass |

### Warm-build bottleneck (documented per SPEC 13)

The warm rebuild still takes ~9 minutes because `LocalGitSource.read_bytes`
runs one `git show` subprocess per file — ~14,800 process spawns dominate
the runtime. The parse+index work itself is seconds at this scale. The fix
(batch reads via `git archive` or a persistent git cat-file process) is a
known optimization, deliberately deferred out of the proof of concept.
The ≤ 5 min warm target is therefore **not met** at corpus scale; the
measured result and bottleneck are recorded here as SPEC 13 requires.
DCLP plus Translations together would take roughly twice as long over the
network (~2× blobs) and was not measured end-to-end in this session.

## Application and search

| Measurement | Result | SPEC 13 target | Status |
|-------------|--------|----------------|--------|
| App startup + artifact validation (14,842 docs, 73.4 MB artifact) | 864 ms | ≤ 5 s | pass |
| Identifier lookup (`TM 23944`, avg of 50) | 0.027 ms | ≤ 100 ms | pass |
| FTS search, polytonic Greek `ἔτους` (avg of 50, BM25 + filters) | 16.4 ms | ≤ 500 ms | pass |
| FTS search, English `horoscope` (avg of 50) | 14.9 ms | ≤ 500 ms | pass |
| First search results without any LLM call | search never contacts the LLM | required | pass |
| Entire SQLite database loaded into memory | no (per-query SQL only) | forbidden | pass |

## Artifact (real `dclp` corpus)

- documents: 14,842
- passages: 13,644
- artifact size: 73.4 MB
- only `DCLP/` blobs were fetched (sparse checkout); the other ~2.8 GB of
  upstream repository content stayed on the server
