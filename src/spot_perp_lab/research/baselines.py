"""Frozen Phase 4 predictor sets and reference predictions."""

from __future__ import annotations

import numpy as np

BASELINE_FEATURES = (
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
)

EXPANDED_ADDITIONS = (
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

EXPANDED_FEATURES = BASELINE_FEATURES + EXPANDED_ADDITIONS

HAC_BASELINE_FEATURES = (
    "spot_log_return_1000ms",
    "spot_log_return_5000ms",
    "spot_quantity_imbalance_1000ms",
    "spot_quantity_imbalance_5000ms",
    "spot_trade_count_1000ms",
    "spot_realised_volatility_5000ms",
)

HAC_EXPANDED_FEATURES = (
    *HAC_BASELINE_FEATURES,
    "perpetual_quantity_imbalance_1000ms",
    "perpetual_quantity_imbalance_5000ms",
    "spot_perp_log_basis",
)


def training_mean_predictions(y_train: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Return the training-target mean reference without using evaluation labels."""

    training_mean = float(np.mean(y_train))
    return np.full(size, training_mean, dtype=np.float64), training_mean
