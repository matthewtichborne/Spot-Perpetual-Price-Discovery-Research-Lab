"""Data-quality and leakage-contract validation for feature frames."""

from __future__ import annotations

import polars as pl


class FeatureValidationError(ValueError):
    """Raised when a feature frame violates its causal schema contract."""


def validate_feature_frame(frame: pl.DataFrame, predictors: list[str], labels: list[str]) -> None:
    """Validate ordering, uniqueness, cutoff timing, and finite numeric values."""

    expected = {"decision_time_ns", "feature_cutoff_ns", *predictors, *labels}
    if set(frame.columns) != expected:
        raise FeatureValidationError(
            f"feature schema mismatch: missing={expected - set(frame.columns)}, "
            f"extra={set(frame.columns) - expected}"
        )
    if frame.height == 0:
        raise FeatureValidationError("feature frame is empty")
    if frame["decision_time_ns"].n_unique() != frame.height:
        raise FeatureValidationError("decision timestamps are not unique")
    if not frame["decision_time_ns"].is_sorted():
        raise FeatureValidationError("decision timestamps are not sorted")
    if frame.filter(pl.col("feature_cutoff_ns") >= pl.col("decision_time_ns")).height:
        raise FeatureValidationError("predictor cutoff is not strictly before decision time")

    numeric_columns = predictors + labels
    non_finite = frame.select(
        pl.sum_horizontal(
            *(
                pl.col(column).is_not_null().and_(~pl.col(column).is_finite()).cast(pl.Int64)
                for column in numeric_columns
            )
        ).sum()
    ).item()
    if non_finite:
        raise FeatureValidationError(f"found {non_finite} non-finite feature or label values")
    imbalance_columns = [column for column in predictors if "_imbalance_" in column]
    invalid_imbalances = frame.select(
        pl.sum_horizontal(
            *(
                ((pl.col(column) < -1.0) | (pl.col(column) > 1.0)).fill_null(False).cast(pl.Int64)
                for column in imbalance_columns
            )
        ).sum()
    ).item()
    if invalid_imbalances:
        raise FeatureValidationError(f"found {invalid_imbalances} flow imbalances outside [-1, 1]")
