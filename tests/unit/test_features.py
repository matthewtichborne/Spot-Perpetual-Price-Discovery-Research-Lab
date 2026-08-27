import math

import polars as pl
import pytest

from spot_perp_lab.config import FeatureConfig
from spot_perp_lab.features.bars import build_base_bars
from spot_perp_lab.features.basis import add_cross_market_features
from spot_perp_lab.features.generate import generate_feature_frame
from spot_perp_lab.features.labels import add_future_return_labels
from spot_perp_lab.features.trade_flow import add_market_features
from spot_perp_lab.features.validate import FeatureValidationError, validate_feature_frame


def _trades(times: list[int], prices: list[float], quantities: list[float]) -> pl.DataFrame:
    signs = [1 if index % 2 == 0 else -1 for index in range(len(times))]
    notionals = [price * quantity for price, quantity in zip(prices, quantities, strict=True)]
    return pl.DataFrame(
        {
            "event_time_ns": times,
            "aggregate_trade_id": list(range(1, len(times) + 1)),
            "price": prices,
            "quantity": quantities,
            "notional": notionals,
            "signed_quantity": [
                sign * quantity for sign, quantity in zip(signs, quantities, strict=True)
            ],
            "signed_notional": [
                sign * notional for sign, notional in zip(signs, notionals, strict=True)
            ],
            "is_buyer_maker": [sign == -1 for sign in signs],
        }
    )


def test_fixed_bar_boundary_is_strictly_causal() -> None:
    trades = _trades([50_000_000, 100_000_000, 150_000_000], [100.0] * 3, [1.0, 2.0, 3.0])
    bars = build_base_bars(trades, 0, 300_000_000, 100)
    assert bars["decision_time_ns"].to_list() == [100_000_000, 200_000_000, 300_000_000]
    assert bars["quantity"].to_list() == [1.0, 5.0, 0.0]
    assert bars["signed_quantity"].to_list() == [1.0, 1.0, 0.0]


def test_hand_calculated_trailing_flow() -> None:
    bars = build_base_bars(
        _trades([50_000_000, 150_000_000], [100.0, 101.0], [2.0, 1.0]),
        0,
        300_000_000,
        100,
    )
    featured, _ = add_market_features(bars, "spot", (100, 200), 100)
    assert featured["spot_signed_quantity_200ms"].to_list() == [2.0, 1.0, -1.0]
    assert featured["spot_quantity_imbalance_200ms"].to_list() == pytest.approx([1.0, 1 / 3, -1.0])


def test_basis_and_future_labels() -> None:
    frame = pl.DataFrame(
        {
            "decision_time_ns": [100, 200, 300],
            "spot_last_price": [100.0, 101.0, 99.0],
            "perpetual_last_price": [101.0, 102.0, 100.0],
            "spot_quantity_100ms": [1.0, 2.0, 1.0],
            "perpetual_quantity_100ms": [2.0, 1.0, 3.0],
            "spot_trade_count_100ms": [1, 2, 1],
            "perpetual_trade_count_100ms": [2, 1, 3],
        }
    )
    with_basis, _ = add_cross_market_features(frame, (100,), 100, 200)
    labelled, labels = add_future_return_labels(with_basis, (100,), 100)
    assert math.isclose(labelled["spot_perp_log_basis"][0], math.log(1.01))
    assert math.isclose(labelled["target_spot_log_return_100ms"][0], math.log(1.01))
    assert labelled["target_spot_direction_100ms"].to_list() == [1, 0, None]
    assert labelled[labels[0]].null_count() == 1


def test_future_perpetual_event_cannot_change_earlier_predictors() -> None:
    config = FeatureConfig(
        base_interval_ms=100,
        decision_interval_ms=500,
        windows_ms=(100, 500),
        label_horizons_ms=(500,),
        basis_z_window_ms=500,
        feature_lag_bars=1,
    )
    times = [50_000_000 + 100_000_000 * index for index in range(19)]
    spot = _trades(times, [100.0 + 0.01 * index for index in range(19)], [1.0] * 19)
    perpetual = _trades(times, [100.1 + 0.01 * index for index in range(19)], [1.0] * 19)
    baseline, predictors, labels = generate_feature_frame(spot, perpetual, 0, 2_000_000_000, config)
    repeated, _, _ = generate_feature_frame(spot, perpetual, 0, 2_000_000_000, config)
    assert baseline.equals(repeated, null_equal=True)
    future = _trades([1_550_000_000], [110.0], [20.0]).with_columns(
        pl.lit(100, dtype=pl.Int64).alias("aggregate_trade_id")
    )
    changed, changed_predictors, _ = generate_feature_frame(
        spot, pl.concat([perpetual, future]), 0, 2_000_000_000, config
    )
    decision = 1_500_000_000
    baseline_row = baseline.filter(pl.col("decision_time_ns") == decision)
    changed_row = changed.filter(pl.col("decision_time_ns") == decision)
    assert predictors == changed_predictors
    assert baseline_row.select(predictors).equals(changed_row.select(predictors), null_equal=True)
    validate_feature_frame(baseline, predictors, labels)


def test_feature_cutoff_must_precede_decision() -> None:
    frame = pl.DataFrame(
        {
            "decision_time_ns": [100],
            "feature_cutoff_ns": [100],
            "predictor": [1.0],
            "target": [0.0],
        }
    )
    with pytest.raises(FeatureValidationError, match="strictly before"):
        validate_feature_frame(frame, ["predictor"], ["target"])
