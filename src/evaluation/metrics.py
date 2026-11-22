"""Metrics on 1-5 star ratings (labels 0-4 are shifted by +1 inside)."""
import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(y_true, y_pred) -> dict:
    """y_true / y_pred: integer labels 0..4."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = np.abs(y_true - y_pred)
    return {
        "accuracy": float((diff == 0).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3, 4])),
        "mae": float(diff.mean()),
        "large_error_rate": float((diff >= 2).mean()),
    }
