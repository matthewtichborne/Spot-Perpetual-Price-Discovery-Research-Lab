"""End-to-end causal feature-frame construction for one symbol-day."""

from __future__ import annotations

import polars as pl

from spot_perp_lab.config import FeatureConfig
from spot_perp_lab.features.bars import build_base_bars
from spot_perp_lab.features.basis import add_cross_market_features
from spot_perp_lab.features.labels import add_future_return_labels
from spot_perp_lab.features.trade_flow import add_market_features


def _market_frame(
    trades: pl.DataFrame,
    prefix: str,
    start_ns: int,
    end_ns: int,
    config: FeatureConfig,
) -> tuple[pl.DataFrame, list[str]]:
    bars = build_base_bars(trades, start_ns, end_ns, config.base_interval_ms)
    featured, feature_names = add_market_features(
        bars, prefix, config.windows_ms, config.base_interval_ms
    )
    raw_columns = [
        column
        for column in featured.columns
        if column != "decision_time_ns" and column not in feature_names
    ]
    return featured.rename({column: f"{prefix}_{column}" for column in raw_columns}), feature_names


def generate_feature_frame(
    spot_trades: pl.DataFrame,
    perpetual_trades: pl.DataFrame,
    start_ns: int,
    end_ns: int,
    config: FeatureConfig,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Construct explicitly lagged predictors and unlagged future labels."""

    spot, spot_features = _market_frame(spot_trades, "spot", start_ns, end_ns, config)
    perpetual, perpetual_features = _market_frame(
        perpetual_trades, "perpetual", start_ns, end_ns, config
    )
    lag = config.feature_lag_bars
    base_interval_ns = config.base_interval_ms * 1_000_000
    decision_interval_ns = config.decision_interval_ms * 1_000_000

    def sample_market(frame: pl.DataFrame, names: list[str]) -> pl.DataFrame:
        return (
            frame.with_columns(*(pl.col(column).shift(lag).alias(column) for column in names))
            .filter(
                ((pl.col("decision_time_ns") - start_ns) % decision_interval_ns == 0)
                & (pl.col("decision_time_ns") < end_ns)
            )
            .select("decision_time_ns", *names)
        )

    spot_sample = sample_market(spot, spot_features)
    perpetual_sample = sample_market(perpetual, perpetual_features)
    spot_cross_columns = ["decision_time_ns", "spot_last_price"] + [
        name
        for window_ms in config.windows_ms
        for name in (
            f"spot_quantity_{window_ms}ms",
            f"spot_trade_count_{window_ms}ms",
        )
    ]
    perpetual_cross_columns = ["decision_time_ns", "perpetual_last_price"] + [
        name
        for window_ms in config.windows_ms
        for name in (
            f"perpetual_quantity_{window_ms}ms",
            f"perpetual_trade_count_{window_ms}ms",
        )
    ]
    cross = spot.select(spot_cross_columns).join(
        perpetual.select(perpetual_cross_columns),
        on="decision_time_ns",
        how="inner",
        validate="1:1",
    )
    cross, cross_features = add_cross_market_features(
        cross, config.windows_ms, config.base_interval_ms, config.basis_z_window_ms
    )
    cross, labels = add_future_return_labels(
        cross, config.label_horizons_ms, config.base_interval_ms
    )
    cross_sample = (
        cross.with_columns(*(pl.col(column).shift(lag).alias(column) for column in cross_features))
        .filter(
            ((pl.col("decision_time_ns") - start_ns) % decision_interval_ns == 0)
            & (pl.col("decision_time_ns") < end_ns)
            & pl.col("spot_last_price").is_not_null()
            & pl.col("perpetual_last_price").is_not_null()
        )
        .select("decision_time_ns", *cross_features, *labels)
    )
    predictors = spot_features + perpetual_features + cross_features
    output = (
        spot_sample.join(perpetual_sample, on="decision_time_ns", validate="1:1")
        .join(cross_sample, on="decision_time_ns", validate="1:1")
        .with_columns(
            (pl.col("decision_time_ns") - lag * base_interval_ns).alias("feature_cutoff_ns")
        )
        .select("decision_time_ns", "feature_cutoff_ns", *predictors, *labels)
        .sort("decision_time_ns")
    )
    return output, predictors, labels
