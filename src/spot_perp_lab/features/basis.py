"""Spot-perpetual basis and relative-activity features."""

from __future__ import annotations

import polars as pl


def add_cross_market_features(
    frame: pl.DataFrame,
    windows_ms: tuple[int, ...],
    base_interval_ms: int,
    basis_z_window_ms: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Add exact-grid cross-market features without forward-looking alignment."""

    basis_name = "spot_perp_log_basis"
    result = frame.with_columns(
        (pl.col("perpetual_last_price") / pl.col("spot_last_price")).log().alias(basis_name)
    )
    feature_names = [basis_name]
    expressions: list[pl.Expr] = []
    for window_ms in windows_ms:
        steps = window_ms // base_interval_ms
        suffix = f"{window_ms}ms"
        basis_change = f"spot_perp_basis_change_{suffix}"
        relative_quantity = f"perpetual_spot_relative_quantity_{suffix}"
        relative_intensity = f"perpetual_spot_relative_intensity_{suffix}"
        expressions.extend(
            [
                (pl.col(basis_name) - pl.col(basis_name).shift(steps)).alias(basis_change),
                pl.when(pl.col(f"spot_quantity_{suffix}") > 0)
                .then(pl.col(f"perpetual_quantity_{suffix}") / pl.col(f"spot_quantity_{suffix}"))
                .otherwise(None)
                .alias(relative_quantity),
                pl.when(pl.col(f"spot_trade_count_{suffix}") > 0)
                .then(
                    pl.col(f"perpetual_trade_count_{suffix}") / pl.col(f"spot_trade_count_{suffix}")
                )
                .otherwise(None)
                .alias(relative_intensity),
            ]
        )
        feature_names.extend([basis_change, relative_quantity, relative_intensity])

    z_steps = basis_z_window_ms // base_interval_ms
    z_name = f"spot_perp_basis_zscore_{basis_z_window_ms}ms"
    rolling_mean = pl.col(basis_name).rolling_mean(z_steps, min_samples=z_steps)
    rolling_std = pl.col(basis_name).rolling_std(z_steps, min_samples=z_steps)
    expressions.append(
        pl.when(rolling_std > 0)
        .then((pl.col(basis_name) - rolling_mean) / rolling_std)
        .otherwise(None)
        .alias(z_name)
    )
    feature_names.append(z_name)
    return result.with_columns(expressions), feature_names
