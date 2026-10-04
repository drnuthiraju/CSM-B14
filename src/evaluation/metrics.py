from __future__ import annotations

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | list[list[int]]]:
    """Compute all reported metrics from actual labels/predictions."""
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0), "f1": f1_score(y_true, y_pred, zero_division=0),
            "hallucination_detection_rate": recall_score(y_true, y_pred, zero_division=0),
            "false_positive_rate": (confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1] / max(1, sum(item == 0 for item in y_true))),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}
