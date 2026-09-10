"""Measure local semantic discovery latency on one artifact.

Usage:
    uv run --extra semantic python scripts/benchmark_discovery.py \
        --artifact data/ddbdp-semantic-v4 --queries 20 --repeats 5

Modes: warm latency (p50/p95) over repeated discovery queries in one process,
plus cold start (fresh interpreter, open, encoder load, first query) measured
by launching this script again with --cold. Timings exclude LLM time by
construction; discovery is fully local.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

QUERIES = (
    "complaints to an official about unpaid wages",
    "registration of a marriage without the archidikastes",
    "payment of taxes in grain",
    "a dispute over a loan repaid in kind",
    "petition to the strategos about property",
    "the sale of a donkey witnessed by friends",
    "a widow asks the epistrategos for help",
    "renting land near the temple of Soknebtunis",
)


def run_warm(artifact: Path, repeats: int, limit: int) -> dict:
    from papyrus_chat.corpus import CorpusService
    from papyrus_chat.retrieval.discovery.models import DiscoveryQuery

    service = CorpusService.open(artifact)
    try:
        info = service.get_corpus_info()
        capability = info.semantic_capability
        timings: list[float] = []
        for _ in range(repeats):
            for text in QUERIES:
                query = DiscoveryQuery(text=text, limit=limit)
                started = time.perf_counter()
                result = service.discover_documents(query)
                elapsed = time.perf_counter() - started
                assert result.available, result.unavailable_reason
                timings.append(elapsed)
        timings.sort()
        return {
            "mode": "warm",
            "artifact": str(artifact),
            "queries_per_repeat": len(QUERIES),
            "repeats": repeats,
            "limit": limit,
            "samples": len(timings),
            "p50_seconds": round(statistics.median(timings), 4),
            "p95_seconds": round(timings[int(len(timings) * 0.95) - 1], 4),
            "max_seconds": round(timings[-1], 4),
            "subjects_available": capability.subjects.available,
            "profiles_count": capability.profiles.count,
            "chunks_count": capability.chunks.count,
        }
    finally:
        service.close()


def run_cold(artifact: Path, limit: int) -> dict:
    from papyrus_chat.corpus import CorpusService
    from papyrus_chat.retrieval.discovery.models import DiscoveryQuery

    started = time.perf_counter()
    service = CorpusService.open(artifact)
    opened = time.perf_counter()
    result = service.discover_documents(DiscoveryQuery(text=QUERIES[0], limit=limit))
    assert result.available, result.unavailable_reason
    first_query = time.perf_counter()
    service.close()
    return {
        "mode": "cold",
        "artifact": str(artifact),
        "open_seconds": round(opened - started, 4),
        "first_query_seconds": round(first_query - opened, 4),
        "total_seconds": round(first_query - started, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--queries", type=int, default=1, help="Query rotation repetitions.")
    parser.add_argument("--repeats", type=int, default=1, help="Rotation repeats for warm mode.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cold", action="store_true", help="Measure cold start and exit.")
    args = parser.parse_args()
    artifact = Path(args.artifact)
    if args.cold:
        print(json.dumps(run_cold(artifact, args.limit)))
        return 0
    if args.queries < 1 or args.repeats < 1:
        print("--queries and --repeats must be positive", file=sys.stderr)
        return 2
    print(json.dumps(run_warm(artifact, args.repeats, args.limit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
