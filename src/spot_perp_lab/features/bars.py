"""Causal fixed-grid aggregation of canonical trade events."""

from __future__ import annotations

import polars as pl

BAR_VALUE_COLUMNS = (
    "quantity",
    "notional",
    "signed_quantity",
    "signed_notional",
    "trade_count",
    "buyer_trade_count",
    "seller_trade_count",
)


def build_base_bars(
    trades: pl.DataFrame,
    start_ns: int,
    end_ns: int,
    interval_ms: int,
) -> pl.DataFrame:
    """Aggregate trades to right-labelled `[start, end)` fixed-grid bars.

    An event is assigned to the first grid boundary strictly after its timestamp.
    Consequently every event in a bar is strictly earlier than the bar label. Empty
    bars have zero activity and carry the last observed price forward.
    """

    interval_ns = interval_ms * 1_000_000
    if end_ns <= start_ns or (end_ns - start_ns) % interval_ns:
        raise ValueError("grid bounds must define a positive whole number of intervals")
    required = {
        "event_time_ns",
        "aggregate_trade_id",
        "price",
        "quantity",
        "notional",
        "signed_quantity",
        "signed_notional",
        "is_buyer_maker",
    }
    if missing := required - set(trades.columns):
        raise ValueError(f"canonical trades are missing columns: {sorted(missing)}")
    if trades.filter(
        (pl.col("event_time_ns") < start_ns) | (pl.col("event_time_ns") >= end_ns)
    ).height:
        raise ValueError("trade timestamp falls outside requested grid")

    grid = pl.DataFrame(
        {
            "decision_time_ns": pl.int_range(
                start_ns + interval_ns,
                end_ns + interval_ns,
                step=interval_ns,
                dtype=pl.Int64,
                eager=True,
            )
        }
    )
    aggregated = (
        trades.sort(["event_time_ns", "aggregate_trade_id"])
        .with_columns(
            (
                ((pl.col("event_time_ns") - start_ns) // interval_ns + 1) * interval_ns + start_ns
            ).alias("decision_time_ns")
        )
        .group_by("decision_time_ns")
        .agg(
            pl.col("price").last().alias("last_price"),
            pl.col("quantity").sum(),
            pl.col("notional").sum(),
            pl.col("signed_quantity").sum(),
            pl.col("signed_notional").sum(),
            pl.len().cast(pl.Int64).alias("trade_count"),
            (~pl.col("is_buyer_maker")).sum().cast(pl.Int64).alias("buyer_trade_count"),
            pl.col("is_buyer_maker").sum().cast(pl.Int64).alias("seller_trade_count"),
        )
    )
    return (
        grid.join(aggregated, on="decision_time_ns", how="left")
        .with_columns(
            pl.col("last_price").forward_fill(),
            *(pl.col(column).fill_null(0) for column in BAR_VALUE_COLUMNS),
        )
        .sort("decision_time_ns")
    )
