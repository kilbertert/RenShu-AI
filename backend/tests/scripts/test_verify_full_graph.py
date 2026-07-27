"""全量图谱验收阈值测试。"""

from scripts.verify_full_graph import evaluate_thresholds


def test_evaluate_thresholds_reports_pass_and_fail():
    results = evaluate_thresholds(
        {"Formula": 25_860, "Herb": 10},
        {"Formula": 25_000, "Herb": 18_000},
    )

    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].actual == 10


def test_evaluate_thresholds_treats_missing_metric_as_zero():
    result = evaluate_thresholds({}, {"Disease": 1})[0]

    assert result.actual == 0
    assert result.passed is False
