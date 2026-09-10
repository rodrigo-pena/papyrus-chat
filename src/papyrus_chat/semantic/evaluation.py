"""Source-linked retrieval judgments and dependency-free ranking metrics.

Unjudged documents are not assumed relevant. These metrics describe recall of
the recorded judgments, not exhaustive thematic recall of the corpus.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RelevanceJudgment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    grade: int = Field(ge=0, le=2)
    source_path: str
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    rationale: str


class EvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    query: str
    query_language: str
    collection: str
    category: str
    judgments: tuple[RelevanceJudgment, ...]

    @property
    def grades(self) -> dict[str, int]:
        return {judgment.document_id: judgment.grade for judgment in self.judgments}


@dataclass(frozen=True)
class RankingMetrics:
    recall: float | None
    ndcg: float | None
    additional_relevant: tuple[str, ...]


def load_cases(path: Path) -> tuple[EvaluationCase, ...]:
    return tuple(EvaluationCase.model_validate(case) for case in json.loads(path.read_text()))


def evaluate_ranking(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    *,
    baseline: Sequence[str] = (),
    k: int = 20,
) -> RankingMetrics:
    """Calculate graded nDCG and judged recall over distinct top-k documents."""
    if k < 1:
        raise ValueError("k must be positive")
    ranked = tuple(dict.fromkeys(ranked_ids))[:k]
    relevant = {document_id for document_id, grade in grades.items() if grade > 0}
    baseline_ids = set(tuple(dict.fromkeys(baseline))[:k])
    ideal = sorted((grade for grade in grades.values() if grade > 0), reverse=True)[:k]

    def dcg(values: Sequence[int]) -> float:
        return sum((2**grade - 1) / log2(rank + 2) for rank, grade in enumerate(values))

    return RankingMetrics(
        recall=len(set(ranked) & relevant) / len(relevant) if relevant else None,
        ndcg=dcg([grades.get(doc, 0) for doc in ranked]) / dcg(ideal) if ideal else None,
        additional_relevant=tuple(doc for doc in ranked if doc in relevant - baseline_ids),
    )
