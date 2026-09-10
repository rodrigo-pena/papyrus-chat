# Semantic retrieval: content indexing, discovery, and evaluation

Semantic content indexing is **opt-in**. A subject-only build remains fully
supported, and every artifact keeps working without content indexes. This page
describes what `--semantic-content` builds, how `discover_documents` ranks
documents, what the local evaluation recorded, and what the feature does not
do.

## What the builder adds

With `--semantic-content` (which always requires `--semantic-model-dir`), the
builder adds two more local float32 indexes beside the subject index, encoded
with the same pinned `multilingual-e5-small` snapshot (384 dimensions):

- **Text chunks** — every nonempty edition and published translation passage is
  split into tokenizer-bounded chunks of 384 content tokens with 64-token
  overlap. Chunks never cross a passage, document, or language boundary, and
  they preserve display text, diacritics, uncertainty markers, character
  offsets, and the passage-level line reference at its actual precision.
- **Document profiles** — one deterministic retrieval text per document, built
  only from the title, descriptive metadata, linked HGV subjects and commentary,
  and representative source excerpts in source order (metadata budget 128
  tokens of 448). Textless documents get an explicitly metadata-only profile.
  Profiles contain no model-generated summaries, translations, or inferred
  themes, and exclude administrative boilerplate.

Encoding is batched, local (FastEmbed ONNX), and independent of the driver
agent's LLM endpoint: builds and queries never contact a network provider.

## Discovery contract

`discover_documents(query)` accepts a natural-language query of up to 500
characters, structural filters (collections, linked-HGV date interval,
transcription languages, passage kinds, passage languages), and a result limit
(default 20, maximum 100). Lexical term groups and HGV subject groups are
explicitly excluded: discovery eligibility never depends on curated labels.

Ranking applies structural scope first, then fuses three channels with
equal-weight reciprocal rank fusion (constant 60, top 200 distinct documents
per channel): profile cosine, best-chunk cosine, and passage BM25 with safely
quoted OR query tokens. Ties break by collection and document ID, documents
contribute at most two matched chunks, and mixed-content profiles are omitted
when passage restrictions apply. The result separates:

- `scope_document_count` and `indexed_document_count` — exact counts of scoped
  documents and of those covered by eligible content indexes; and
- `hits` — ranked candidates with contributing channels, channel scores, chunk
  locations, and canonical URLs.

Ranked candidates are never a thematic match count, and discovery results say
so in their `method` field. Chunk locations open directly in
`inspect_documents` via the returned `chunk_ids`, so any substantive textual
claim still comes from a real excerpt with source provenance.

`get_corpus_info` reports subject/profile/chunk capabilities separately, each
with its own count and `unavailable_reason`. An artifact without content
indexes, or a runtime without the `[semantic]` extra, makes discovery explicitly
unavailable rather than returning empty results, so callers can disclose the
limitation instead of inferring absence.

## Retrieval evaluation

The evaluation set is the 30 source-linked, agent-curated questions in
`tests/fixtures/semantic/questions.json` (judgments with source paths, commit
provenance, rationales, and grade-0 hard negatives) recorded against the five
pinned fixture documents; `tests/fixtures/semantic/README.md` records its
known limits. Each question is ranked by the shipped retrieval paths at
Recall@20 / nDCG@20 with graded relevance and unjudged-scored-zero scoring:

- **baseline** — the reproducible lexical baseline in `baseline.json` (one OR
  term group of query words across all fields, no collection or subject
  restrictions). It is a floor, not the full agent baseline: an LLM composing
  multilingual synonyms or choosing HGV labels can do better.
- **profiles** — profile cosine only.
- **chunks** — best-chunk cosine only.
- **fusion** — the shipped three-channel discovery used by the agent and MCP.

Reproduce with:

```bash
uv run --extra semantic python scripts/evaluate_semantic_retrieval.py \
  --artifact data/fixture-semantic \
  --output tests/fixtures/semantic/results-fixture.json
```

Results with the real pinned model on an Apple M5 Max (2026-09-10):

| Variant  | Recall@20 | nDCG@20 | Relevant docs beyond baseline |
| -------- | --------- | ------- | ----------------------------- |
| baseline | 0.833     | 0.683   | — (by construction)           |
| profiles | 1.000     | 0.900   | 5                             |
| chunks   | 0.933     | 0.831   | 5                             |
| fusion   | 1.000     | 0.756   | 5                             |

Per collection (n = 10 ddbdp, 8 dclp, 12 translations):

| Collection   | baseline R@20 / n@20 | profiles R@20 / n@20 | chunks R@20 / n@20 | fusion R@20 / n@20 |
| ------------ | -------------------- | -------------------- | ------------------ | ------------------ |
| ddbdp        | 0.900 / 0.689        | 1.000 / 0.749        | 1.000 / 0.729      | 1.000 / 0.565      |
| dclp         | 0.625 / 0.462        | 1.000 / 1.000        | 0.750 / 0.750      | 1.000 / 0.676      |
| translations | 0.917 / 0.824        | 1.000 / 0.958        | 1.000 / 0.969      | 1.000 / 0.969      |

Reading of the smoke result:

- The lexical baseline missed judged-relevant documents entirely on 5 of 30
  questions: a French query about greetings between a father and son, an
  English and a German horoscope query about planetary degrees, an English
  query about fractional astronomical degrees, and a German query about daily
  maintenance for boat crews. Profile and chunk channels found all of them,
  and the shipped fusion returns all 5 in its top-20 — the concrete
  "additional relevant evidence" this feature adds.
- Fusion's nDCG is below the strongest single channel because equal-weight RRF
  trades exact top-ordering for channel complementarity; the baseline keeps
  better ordering when its term group happens to match the judgment. On the
  five-document fixture this ordering penalty is large and expected.
- Recall@20 on a five-document fixture is permissive; these numbers measure
  smoke behavior, not corpus-scale ranking quality. Judgments have not been
  independently reviewed by a papyrologist.

Content indexing stays opt-in until evaluation on a larger, independently
reviewed judgment set demonstrates added relevant evidence beyond this smoke
signal.

## Local performance

Measured on an Apple M5 Max (128 GB RAM), macOS, Python 3.13, FastEmbed ONNX
with the pinned `multilingual-e5-small` revision, full DDbDP artifact
(67,980 documents / 95,902 passages from upstream commit `ffc23d0174e8`).
Reproduce the build and measurements with:

```bash
uv sync --extra semantic
/usr/bin/time -l uv run papyrus-corpus-build ddbdp \
  --ref ffc23d0174e810ff338bd1048ed0e5882a816fdc \
  --semantic-model-dir models/multilingual-e5-small \
  --semantic-content \
  --output data/ddbdp-semantic-v4
uv run python scripts/benchmark_discovery.py --artifact data/ddbdp-semantic-v4 \
  --repeats 10   # warm p50/p95
uv run python scripts/benchmark_discovery.py --artifact data/ddbdp-semantic-v4 --cold
```

| Measurement                                | Result                                | Target     | Status |
| ------------------------------------------ | ------------------------------------- | ---------- | ------ |
| Subject-only build (same builder, no content) | 109.2 s                            | ≤ 2 min    | pass   |
| Full content build (chunks + profiles)     | 5,587.8 s (1 h 33 min)                | observe    | —      |
| Chunk encoding (120,215 chunks)            | ~54 min 40 s (≈ 37 chunks/s)          | observe    | —      |
| Profile encoding (67,980 profiles)         | ~36 min 30 s (≈ 31 profiles/s)        | observe    | —      |
| Artifact size, subject-only                | 2.62 GB                               | observe    | —      |
| Artifact size, with content indexes        | 3.04 GB (+425 MB, ≈ +16%)             | observe    | —      |
| Content float32 files                      | chunks 192 MB + profiles 112 MB       | observe    | —      |
| Peak memory during content build           | ≈ 9.6 GB                              | observe    | —      |
| Cold start (interpreter → first discovery result) | 3.2 s (2.5 s from service open; encoder init dominates) | ≤ 5 s | pass |
| Warm discovery p50 (80 samples)            | 0.325 s                               | observe    | —      |
| Warm discovery p95                         | 0.344 s                               | < 2 s      | pass   |

Notes:

- The content build is roughly 51 times the subject-only build; chunk and
  profile encoding are the cost, not parsing or SQLite writes. The encoder
  used about 5 GB resident steady-state with multi-threaded ONNX.
- The artifact bundles one model snapshot (~1.6 GB) shared by subjects,
  profiles, and chunks; the float32 vector files for content add ~304 MB, and
  the remaining growth is chunk/profile rows in `corpus.sqlite`.
- Warm p95 of 0.34 s includes structural scope SQL, the mmap-backed cosine
  scans over both matrices, BM25, and reciprocal-rank fusion, with no LLM
  involved. The p95 target is met with a 6x margin.
- First-query latency is dominated by loading the local encoder; later
  queries reuse it. Subject suggestions and discovery share that one encoder.

## Rebuild requirements

- Content indexes are part of the manifest's semantic section with their own
  counts, files, preprocessing versions, and hashes; schema v4 validates
  vector-row alignment, finite vectors, offsets, and file integrity.
- Rebuilding with different preprocessing versions or a different model
  snapshot produces a different artifact; validation marks such artifacts
  incompatible rather than silently loading them. The existing rebuild-required
  policy for older artifacts is unchanged, and a rebuild must follow the
  stop-server/rebuild/restart sequence in [local MCP](local-mcp.md).
- One shared bundled model covers subject suggestions, profiles, and chunks;
  there is no per-collection model choice.

## Known limitations and follow-up work

- Discovery is scoped to one artifact; documents are never merged across
  collections. Cross-collection merging of identical texts is follow-up work.
- RRF fusion weights are fixed and equal; weighted or learned fusion, model
  comparison, and generated enrichment remain follow-up work.
- Profiles and chunks rank with the same model as queries (symmetric e5 usage
  with asymmetric prefixes); no re-ranker is applied.
- Warm-query latency is measured without any LLM involvement; the p95 target
  below two seconds applies to the retrieval call only.
