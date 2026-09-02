# Performance measurements

The baseline measurements were recorded on 2026-08-31 on an Apple M5 Max with
128 GB RAM, macOS, Python 3.13, and a broadband connection. The schema-v2
measurements were recorded on the same machine on 2026-09-01. The full-corpus
baseline predates the DDbDP/HGV implementation and is retained as a reference.

## Full-corpus baseline

The measured baseline built the real `dclp` collection with the v0.1.0
builder. It covered 14,842 documents and produced a 73.4 MB schema-v1
artifact.

| Measurement                                         | Result                        | Target   | Status |
| --------------------------------------------------- | ----------------------------- | -------- | ------ |
| Remote `dclp` build, cold cache                     | 565-608 s across two runs     | ≤ 15 min | pass   |
| Remote `dclp` rebuild, warm cache                   | 545 s                         | ≤ 5 min  | miss   |
| Artifact startup and validation                     | 864 ms                        | ≤ 5 s    | pass   |
| Identifier lookup (`TM 23944`, average of 50)       | 0.027 ms                      | ≤ 100 ms | pass   |
| FTS search, polytonic Greek `ἔτους` (average of 50) | 16.4 ms                       | ≤ 500 ms | pass   |
| FTS search, English `horoscope` (average of 50)     | 14.9 ms                       | ≤ 500 ms | pass   |
| First search without an LLM call                    | Search never contacts the LLM | required | pass   |

The warm rebuild bottleneck was one `git show` subprocess per file. Remote
builds now keep one `git cat-file --batch` process open after sparse checkout,
while local checkout builds retain the simpler per-file object read because
they are primarily a development path.

## Schema-v2 DDbDP/HGV rebuild

These measurements use upstream commit `ffc23d0174e8`, a warm source cache,
and the same machine as the baseline. The failing run is the user-reported
pre-change build. It reached its first SQLite uniqueness error only after both
collections had parsed. The verification run includes duplicate-value
normalization, a complete pre-write record-graph audit, and bulk document FTS
indexing.

| Measurement                          | Failing run | Verification run | Change |
| ------------------------------------ | ----------- | ---------------- | ------ |
| Parse 67,980 DDbDP records           | 1,344.6 s   | 387.9 s          | -71%   |
| Parse/link 66,872 HGV records        | 1,050.7 s   | 257.9 s          | -75%   |
| Combined parse and link              | 2,395.3 s   | 645.8 s          | -73%   |
| Pre-write normalized integrity audit | absent      | <1 s             | added  |
| SQLite write                         | failed      | 10 s             | pass   |
| Complete artifact build              | failed      | 665.0 s          | pass   |

The successful artifact contains 67,980 documents, 95,902 passages, and
134,653 source components in an 872,867,224-byte artifact. The pre-change
writer also refreshed document FTS with several queries per document. The
bulk writer replaces that N+1 loop with set-based aggregation and insertion.

A focused warm-cache benchmark of 1,000 real HGV object reads measured the
old per-file `git show` path at 10.589 s and the final persistent exact-object
reader at 0.064 s (about 165 times faster at the read layer). The full
verification run used the same no-per-file-process strategy; the reader was
then hardened to use commit objects rather than checked-out file bytes so a
dirty or concurrently changed cache cannot affect reproducibility.

## Schema-v3 DDbDP semantic build

An observed build on an Apple M5 Max MacBook Pro bundled the local multilingual
embedding model and generated semantic subject vectors with the following
command:

```console
uv run papyrus-corpus-build ddbdp \
  --semantic-model-dir ./models/multilingual-e5-small \
  --output ./data/ddbdp
```

| Measurement                       | Result                         | Target  | Status |
| --------------------------------- | ------------------------------ | ------- | ------ |
| DDbDP build with semantic vectors | 115.36 s (1 min 55 s)          | ≤ 2 min | pass   |
| Resulting artifact size           | 2,616,737,301 bytes (~2.62 GB) | observe | —      |

The artifact-size increase is expected: the portable semantic model snapshot
is bundled alongside the corpus, subject vocabulary, and float32 embeddings.

## v2 validation

The automated coverage builds the paired real-source DDbDP/HGV fixture,
validates schema-v2 tables and links, and exercises distinct-document FTS,
date overlap, language filters, facets, and bounded evidence. The full pinned
DDbDP/HGV build above additionally verifies the normalized integrity audit,
artifact validation, and exact duplicate metadata fix against upstream data.

## Acceptance feedback

No live collaborator conversation or feedback session was available during
this implementation run. The four-query walkthrough is documented in the
README and covered at the contract level by deterministic `FunctionModel`
dialogue evaluations; it still needs a human smoke run against a configured
tool-calling provider.
