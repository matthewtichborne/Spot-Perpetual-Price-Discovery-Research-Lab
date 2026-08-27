"""Trailing same-market return, activity, flow, and volatility features."""

from __future__ import annotations

import polars as pl

from spot_perp_lab.features.volatility import realised_volatility


def add_market_features(
    bars: pl.DataFrame,
    prefix: str,
    windows_ms: tuple[int, ...],
    base_interval_ms: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Add trailing-window predictors using only bars at or before each row."""

    frame = bars.with_columns(pl.col("last_price").log().diff().alias("base_log_return"))
    feature_names: list[str] = []
    expressions: list[pl.Expr] = []
    for window_ms in windows_ms:
        steps = window_ms // base_interval_ms
        suffix = f"{window_ms}ms"
        quantity = pl.col("quantity").rolling_sum(steps, min_samples=1)
        notional = pl.col("notional").rolling_sum(steps, min_samples=1)
        signed_quantity = pl.col("signed_quantity").rolling_sum(steps, min_samples=1)
        signed_notional = pl.col("signed_notional").rolling_sum(steps, min_samples=1)
        trades = pl.col("trade_count").rolling_sum(steps, min_samples=1)
        buyer_trades = pl.col("buyer_trade_count").rolling_sum(steps, min_samples=1)
        seller_trades = pl.col("seller_trade_count").rolling_sum(steps, min_samples=1)
        volatility = realised_volatility("base_log_return", steps)
        names = {
            "signed_quantity": f"{prefix}_signed_quantity_{suffix}",
            "signed_notional": f"{prefix}_signed_notional_{suffix}",
            "quantity": f"{prefix}_quantity_{suffix}",
            "notional": f"{prefix}_notional_{suffix}",
            "quantity_imbalance": f"{prefix}_quantity_imbalance_{suffix}",
            "notional_imbalance": f"{prefix}_notional_imbalance_{suffix}",
            "buyer_trades": f"{prefix}_buyer_trades_{suffix}",
            "seller_trades": f"{prefix}_seller_trades_{suffix}",
            "trade_count": f"{prefix}_trade_count_{suffix}",
            "arrival_intensity": f"{prefix}_arrival_intensity_{suffix}",
            "average_trade_size": f"{prefix}_average_trade_size_{suffix}",
            "log_return": f"{prefix}_log_return_{suffix}",
            "realised_volatility": f"{prefix}_realised_volatility_{suffix}",
            "flow_volatility_interaction": f"{prefix}_flow_volatility_interaction_{suffix}",
        }
        quantity_imbalance = (
            pl.when(trades > 0).then((signed_quantity / quantity).clip(-1.0, 1.0)).otherwise(0.0)
        )
        notional_imbalance = (
            pl.when(trades > 0).then((signed_notional / notional).clip(-1.0, 1.0)).otherwise(0.0)
        )
        expressions.extend(
            [
                signed_quantity.alias(names["signed_quantity"]),
                signed_notional.alias(names["signed_notional"]),
                quantity.alias(names["quantity"]),
                notional.alias(names["notional"]),
                quantity_imbalance.alias(names["quantity_imbalance"]),
                notional_imbalance.alias(names["notional_imbalance"]),
                buyer_trades.alias(names["buyer_trades"]),
                seller_trades.alias(names["seller_trades"]),
                trades.alias(names["trade_count"]),
                (trades / (window_ms / 1_000)).alias(names["arrival_intensity"]),
                pl.when(trades > 0)
                .then(quantity / trades)
                .otherwise(0.0)
                .alias(names["average_trade_size"]),
                (pl.col("last_price").log() - pl.col("last_price").shift(steps).log()).alias(
                    names["log_return"]
                ),
                volatility.alias(names["realised_volatility"]),
                (quantity_imbalance * volatility).alias(names["flow_volatility_interaction"]),
            ]
        )
        feature_names.extend(names.values())
    return frame.with_columns(expressions).drop("base_log_return"), feature_names
