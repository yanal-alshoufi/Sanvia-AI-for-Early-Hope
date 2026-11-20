import math

import numpy as np

from sanvia.evaluation.metrics import compute_task_metrics, selection_score


def test_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    m = compute_task_metrics(y, y, np.eye(5)[y], 5)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["qwk"] == 1.0 and m["macro_auc"] == 1.0


def test_absent_classes_are_nan_not_zero():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    prob = np.eye(5)[y_pred]
    m = compute_task_metrics(y_true, y_pred, prob, 5)
    assert math.isnan(m["per_class"]["recall"][3]) and math.isnan(m["per_class"]["f1"][4])
    assert m["classes_missing_in_truth"] == [2, 3, 4]
    assert abs(m["macro_recall"] - 0.75) < 1e-9 and m["notes"]


def test_unpredicted_class_has_nan_precision_zero_f1():
    m = compute_task_metrics(np.array([0, 1, 1]), np.array([0, 0, 0]), np.eye(2)[[0, 0, 0]], 2)
    assert math.isnan(m["per_class"]["precision"][1]) and m["per_class"]["f1"][1] == 0.0


def test_qwk_penalises_distant_errors_more():
    y = np.array([0, 1, 2, 3, 4] * 4)
    near = (y + 1) % 5
    near[y == 4] = 3
    far = 4 - y
    q_near = compute_task_metrics(y, near, np.eye(5)[near], 5)["qwk"]
    q_far = compute_task_metrics(y, far, np.eye(5)[far], 5)["qwk"]
    assert q_near > q_far


def test_empty_input_all_nan():
    m = compute_task_metrics(np.array([]), np.array([]), np.zeros((0, 4)), 4)
    assert math.isnan(m["macro_f1"]) and math.isnan(m["accuracy"])


def test_selection_score():
    assert abs(selection_score(0.4, 0.8) - 0.6) < 1e-12 and math.isnan(selection_score(math.nan, 0.5))
