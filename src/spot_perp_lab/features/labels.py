"""Future spot-return targets kept separate from predictor construction."""

from __future__ import annotations

import polars as pl


def add_future_return_labels(
    frame: pl.DataFrame,
    horizons_ms: tuple[int, ...],
    base_interval_ms: int,
    price_column: str = "spot_last_price",
) -> tuple[pl.DataFrame, list[str]]:
    """Add continuous and directional future spot-return labels."""

    expressions: list[pl.Expr] = []
    label_names: list[str] = []
    for horizon_ms in horizons_ms:
        steps = horizon_ms // base_interval_ms
        return_name = f"target_spot_log_return_{horizon_ms}ms"
        direction_name = f"target_spot_direction_{horizon_ms}ms"
        future_return = pl.col(price_column).shift(-steps).log() - pl.col(price_column).log()
        expressions.extend(
            [
                future_return.alias(return_name),
                pl.when(future_return.is_null())
                .then(None)
                .otherwise((future_return > 0).cast(pl.Int8))
                .alias(direction_name),
            ]
        )
        label_names.extend([return_name, direction_name])
    return frame.with_columns(expressions), label_names
