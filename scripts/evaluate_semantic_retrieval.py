"""Run the semantic retrieval evaluation over one semantic-content artifact.

Usage:
    uv run python scripts/evaluate_semantic_retrieval.py --artifact data/fixture-semantic

For every case in tests/fixtures/semantic/questions.json, the script ranks all
artifact documents with the recorded lexical baseline, profile cosine, best
chunk cosine, and the shipped three-channel fusion, then reports Recall@20,
nDCG@20, and relevant documents found beyond the baseline, with
collection/language breakdowns. Results are written as JSON and printed as a
markdown summary. Metrics describe recall of the recorded judgments only;
unjudged documents are scored zero, not asserted irrelevant.
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from papyrus_chat.semantic.evaluation import evaluate_ranking, load_cases

CHANNELS = ("profiles", "chunks", "lexical")
VARIANTS = {
    "baseline": "Recorded lexical baseline (OR term group, all fields).",
    "profiles": "Profile cosine only.",
    "chunks": "Best-chunk cosine only.",
    "fusion": "Shipped discovery fusion (profiles, chunks, lexical).",
}


def build_fixture_artifact(output: Path) -> None:
    """Build a real-model semantic-content artifact from the fixture corpus."""
    from papyrus_chat.builder.pipeline import build_artifact
    from papyrus_chat.builder.source import LocalGitSource

    fixtures = Path("tests/fixtures/idp.data")
    repo = Path(tempfile.mkdtemp(prefix="papyrus-eval-git-")) / "idp.data"
    import shutil
    import subprocess

    shutil.copytree(fixtures, repo)
    for command in (
        ["git", "init", "-b", "master"],
        ["git", "config", "user.email", "eval@example.com"],
        ["git", "config", "user.name", "Evaluation"],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture snapshot"],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    build_artifact(
        ["ddbdp", "dclp", "translations"],
        output=output,
        source=LocalGitSource(repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
        semantic_model_dir=Path("models/multilingual-e5-small"),
        semantic_content=True,
    )


def rank(service, case_query: str, variant: str) -> tuple[str, ...]:
    from papyrus_chat.retrieval.discovery.models import DiscoveryQuery

    query = DiscoveryQuery(text=case_query, limit=20)
    if variant == "fusion":
        result = service.discover_documents(query)
    else:
        channels = (variant,) if variant in CHANNELS else None
        if channels is None:
            raise ValueError(f"unknown variant {variant}")
        result = service._search.discovery.search(query, channels=channels)  # noqa: SLF001
    return tuple(hit.document_id for hit in result.hits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Semantic-content artifact directory.")
    parser.add_argument(
        "--cases", default="tests/fixtures/semantic/questions.json", help="Evaluation cases JSON."
    )
    parser.add_argument(
        "--baseline", default="tests/fixtures/semantic/baseline.json", help="Baseline JSON."
    )
    parser.add_argument("--output", help="Write full JSON results to this path.")
    args = parser.parse_args()

    artifact = Path(args.artifact)
    if not artifact.is_dir():
        build_fixture_artifact(artifact)
        print(f"Built evaluation artifact: {artifact}", file=sys.stderr)

    from papyrus_chat.corpus import CorpusService

    service = CorpusService.open(artifact)
    try:
        cases = load_cases(Path(args.cases))
        baseline = {
            entry["case_id"]: entry for entry in json.loads(Path(args.baseline).read_text())
        }
        results: dict[str, dict] = {}
        per_variant: dict[str, list] = defaultdict(list)
        per_language: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for case in cases:
            grades = case.grades
            case_rows = {}
            for variant in VARIANTS:
                ranked = (
                    baseline[case.case_id]["ranked_ids"]
                    if variant == "baseline"
                    else rank(service, case.query, variant)
                )
                baseline_ids = baseline[case.case_id]["ranked_ids"]
                metrics = evaluate_ranking(ranked, grades, baseline=baseline_ids, k=20)
                case_rows[variant] = {
                    "ranked_ids": ranked,
                    "recall": metrics.recall,
                    "ndcg": metrics.ndcg,
                    "additional_relevant": metrics.additional_relevant,
                }
                if metrics.recall is not None:
                    per_variant[variant].append(metrics)
                    per_language[case.collection][variant].append(metrics)
            results[case.case_id] = {
                "query": case.query,
                "query_language": case.query_language,
                "collection": case.collection,
                "category": case.category,
                **case_rows,
            }

        def summarize(metrics_list) -> dict:
            recalls = [m.recall for m in metrics_list]
            ndcgs = [m.ndcg for m in metrics_list]
            extra = sum(len(m.additional_relevant) for m in metrics_list)
            n = len(metrics_list)
            return {
                "cases": n,
                "recall@20": round(sum(recalls) / n, 4),
                "ndcg@20": round(sum(ndcgs) / n, 4),
                "additional_relevant_docs": extra,
            }

        summary = {variant: summarize(rows) for variant, rows in per_variant.items()}
        by_collection = {
            collection: {variant: summarize(rows) for variant, rows in variants.items()}
            for collection, variants in sorted(per_language.items())
        }
        output = {"summary": summary, "by_collection": by_collection, "cases": results}
        if args.output:
            Path(args.output).write_text(json.dumps(output, indent=1, ensure_ascii=False))
        print(json.dumps({"summary": summary, "by_collection": by_collection}, indent=1))
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
