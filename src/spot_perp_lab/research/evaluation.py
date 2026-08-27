"""Out-of-sample predictive metrics and daily breakdowns."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, training_mean: float
) -> dict[str, float]:
    """Compute regression metrics with training-mean-referenced OOS R-squared."""

    errors = actual - predicted
    reference_errors = actual - training_mean
    reference_sse = float(reference_errors @ reference_errors)
    model_sse = float(errors @ errors)
    return {
        "oos_r2": 1.0 - model_sse / reference_sse if reference_sse > 0 else 0.0,
        "mse": float(mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "pearson_ic": _correlation(actual, predicted),
        "rank_ic": _correlation(rankdata(actual), rankdata(predicted)),
    }


def classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute probability and threshold classification metrics."""

    predicted = (probability >= 0.5).astype(np.int8)
    classes = np.unique(actual)
    return {
        "roc_auc": float(roc_auc_score(actual, probability)) if classes.size == 2 else 0.0,
        "pr_auc": float(average_precision_score(actual, probability)),
        "brier": float(brier_score_loss(actual, probability)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "class_balance": float(np.mean(actual)),
    }
