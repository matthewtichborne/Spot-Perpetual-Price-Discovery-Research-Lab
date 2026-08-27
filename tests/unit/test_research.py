from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from spot_perp_lab.research.evaluation import regression_metrics
from spot_perp_lab.research.inference import hac_regression, paired_day_bootstrap
from spot_perp_lab.research.models import regression_pipeline
from spot_perp_lab.research.phase5 import (
    PLACEBO_SHIFT_ROWS,
    XGBOOST_GRID,
    _circular_placebo_matrix,
    _xgboost_tuning_masks,
)
from spot_perp_lab.research.splits import expanding_day_folds, split_frame


def test_expanding_folds_are_chronological_and_disjoint() -> None:
    start = date(2025, 1, 2)
    dates = tuple((start + timedelta(days=offset)).isoformat() for offset in range(20))
    folds = expanding_day_folds(dates, initial_train_days=10, evaluation_days=5)
    assert len(folds) == 2
    assert folds[0].train_dates == dates[:10]
    assert folds[0].evaluation_dates == dates[10:15]
    assert folds[1].train_dates == dates[:15]
    assert folds[1].evaluation_dates == dates[15:20]
    assert not set(folds[0].train_dates) & set(folds[0].evaluation_dates)


def test_split_applies_boundary_purge_and_embargo() -> None:
    boundary = 1_735_776_000_000_000_000
    frame = pl.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
            "decision_time_ns": [
                boundary - 11_000_000_000,
                boundary - 5_000_000_000,
                boundary + 5_000_000_000,
                boundary + 11_000_000_000,
            ],
        }
    )
    fold = expanding_day_folds(("2025-01-01", "2025-01-02"), 1, 1)[0]
    train, evaluation = split_frame(frame, fold, purge_seconds=10)
    assert train["decision_time_ns"].to_list() == [boundary - 11_000_000_000]
    assert evaluation["decision_time_ns"].to_list() == [boundary + 11_000_000_000]


def test_preprocessing_statistics_use_training_only() -> None:
    estimator = regression_pipeline("ridge", ridge_alpha=1.0)
    x_train = np.array([[0.0], [1.0], [np.nan]])
    y_train = np.array([0.0, 1.0, 0.5])
    estimator.fit(x_train, y_train)
    estimator.predict(np.array([[1_000.0], [np.nan]]))
    assert estimator.named_steps["imputer"].statistics_[0] == 0.5
    assert estimator.named_steps["scaler"].mean_[0] == 0.5


def test_oos_r2_uses_training_mean_reference() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(actual, np.array([1.0, 2.0, 3.0]), training_mean=0.0)
    assert metrics["oos_r2"] == 1.0
    mean_metrics = regression_metrics(actual, np.zeros(3), training_mean=0.0)
    assert mean_metrics["oos_r2"] == 0.0


def test_day_bootstrap_is_deterministic() -> None:
    improvements = np.array([1.0, 2.0, -0.5, 0.25])
    first = paired_day_bootstrap(improvements, replicates=500, random_seed=7)
    second = paired_day_bootstrap(improvements, replicates=500, random_seed=7)
    assert first == second
    assert first.lower <= first.estimate <= first.upper


def test_hac_inference_returns_every_term() -> None:
    generator = np.random.default_rng(4)
    features = generator.normal(size=(200, 2))
    target = 0.2 * features[:, 0] + generator.normal(scale=0.1, size=200)
    rows = hac_regression(features, target, ("first", "second"), max_lags=3, scope="test")
    assert [row["term"] for row in rows] == ["intercept", "first", "second"]
    assert all(row["observations"] == 200 for row in rows)


def test_too_short_date_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        expanding_day_folds(("2025-01-01", "2025-01-02"), 2, 1)


def test_phase5_xgboost_grid_is_deliberately_bounded() -> None:
    assert len(XGBOOST_GRID) == 4
    assert {row["max_depth"] for row in XGBOOST_GRID} == {2, 3}
    assert {row["n_estimators"] for row in XGBOOST_GRID} == {100, 200}


def test_phase5_tuning_split_uses_january_only_and_five_second_training_rows() -> None:
    boundary = int(datetime(2025, 1, 27, tzinfo=UTC).timestamp()) * 1_000_000_000
    frame = pl.DataFrame(
        {
            "decision_time_ns": [
                boundary - 15_000_000_000,
                boundary - 14_000_000_000,
                boundary + 5_000_000_000,
                boundary + 15_000_000_000,
            ],
            "target_spot_log_return_5000ms": [0.0, 0.0, 0.0, 0.0],
        }
    )
    train, validation = _xgboost_tuning_masks(frame)
    assert train["decision_time_ns"].to_list() == [boundary - 15_000_000_000]
    assert validation["decision_time_ns"].to_list() == [boundary + 15_000_000_000]


def test_placebo_shift_is_circular_within_each_day() -> None:
    rows = PLACEBO_SHIFT_ROWS + 3
    frame = pl.DataFrame(
        {
            "date": ["2025-01-01"] * rows,
            **{
                name: np.arange(rows, dtype=float)
                if name == "perpetual_log_return_1000ms"
                else np.zeros(rows)
                for name in (
                    "spot_log_return_1000ms",
                    "spot_log_return_5000ms",
                    "spot_quantity_imbalance_1000ms",
                    "spot_quantity_imbalance_5000ms",
                    "spot_signed_notional_1000ms",
                    "spot_signed_notional_5000ms",
                    "spot_trade_count_1000ms",
                    "spot_trade_count_5000ms",
                    "spot_notional_1000ms",
                    "spot_notional_5000ms",
                    "spot_realised_volatility_1000ms",
                    "spot_realised_volatility_5000ms",
                    "perpetual_log_return_1000ms",
                    "perpetual_log_return_5000ms",
                    "perpetual_quantity_imbalance_1000ms",
                    "perpetual_quantity_imbalance_5000ms",
                    "perpetual_signed_notional_1000ms",
                    "perpetual_signed_notional_5000ms",
                    "perpetual_trade_count_1000ms",
                    "perpetual_trade_count_5000ms",
                    "perpetual_realised_volatility_1000ms",
                    "perpetual_realised_volatility_5000ms",
                    "spot_perp_log_basis",
                    "spot_perp_basis_change_1000ms",
                    "spot_perp_basis_zscore_10000ms",
                    "perpetual_spot_relative_quantity_1000ms",
                    "perpetual_spot_relative_intensity_1000ms",
                )
            },
        }
    )
    matrix = _circular_placebo_matrix(frame)
    first_addition = 12
    assert matrix[PLACEBO_SHIFT_ROWS, first_addition] == 0.0
    assert matrix[0, first_addition] == 3.0
