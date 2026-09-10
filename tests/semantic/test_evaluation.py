from pathlib import Path

import pytest

from papyrus_chat.semantic.evaluation import evaluate_ranking, load_cases


def test_metrics_use_unique_results_and_graded_relevance() -> None:
    result = evaluate_ranking(["b", "b", "x", "a"], {"a": 2, "b": 1}, baseline=["a"], k=3)
    assert result.recall == 1.0
    assert result.ndcg == pytest.approx(2.5 / (3 + 1 / 1.584962500721156))
    assert result.additional_relevant == ("b",)


def test_empty_judgments_are_not_treated_as_perfect_recall() -> None:
    result = evaluate_ranking(["a"], {}, k=20)
    assert result.recall is None
    assert result.ndcg is None


def test_cases_cover_sources_languages_and_hard_negatives() -> None:
    cases = load_cases(Path("tests/fixtures/semantic/questions.json"))
    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == 30
    assert {case.collection for case in cases} == {"ddbdp", "dclp", "translations"}
    assert {case.query_language for case in cases} >= {"en", "fr", "de", "grc"}
    for case in cases:
        assert case.judgments
        assert any(j.grade == 0 for j in case.judgments)
        assert any(j.grade > 0 for j in case.judgments)
        for judgment in case.judgments:
            assert len(judgment.source_commit) == 40
            assert (Path("tests/fixtures/idp.data") / judgment.source_path).is_file()
            assert judgment.rationale
