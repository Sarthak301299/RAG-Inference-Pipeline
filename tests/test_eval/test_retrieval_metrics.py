import pytest

from src.eval.retrieval_metrics import average_metric, mrr, precision_at_k, recall_at_k


def test_precision_at_k_perfect_match():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "b", "c"]

    assert precision_at_k(retrieved, relevant, 3) == 1.0


def test_precision_at_k_partial_match():
    retrieved = ["a", "x", "c"]
    relevant = ["a", "b", "c"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_precision_at_k_no_match():
    retrieved = ["x", "y", "z"]
    relevant = ["a", "b", "c"]

    assert precision_at_k(retrieved, relevant, 3) == 0.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], ["a", "b"], 3) == 0.0


def test_precision_at_k_k_larger_than_retrieved():
    retrieved = ["a"]
    relevant = ["a", "b"]

    assert precision_at_k(retrieved, relevant, 5) == 1.0


def test_precision_at_k_invalid_k_raises():
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], 0)


def test_recall_at_k_perfect_match():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "b"]

    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_recall_at_k_partial_match():
    retrieved = ["a", "x", "c"]
    relevant = ["a", "b", "c"]

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k_no_relevant_ids():
    assert recall_at_k(["a", "b"], [], 3) == 0.0


def test_recall_at_k_k_smaller_than_relevant():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "b", "c", "d"]

    assert recall_at_k(retrieved, relevant, 1) == pytest.approx(1 / 4)


def test_recall_at_k_invalid_k_raises():
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], -1)


def test_mrr_first_result_relevant():
    assert mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_third_result_relevant():
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_mrr_no_relevant_found():
    assert mrr(["x", "y", "z"], ["a"]) == 0.0


def test_mrr_empty_retrieved():
    assert mrr([], ["a"]) == 0.0


def test_average_metric_normal():
    assert average_metric([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_average_metric_empty():
    assert average_metric([]) == 0.0
