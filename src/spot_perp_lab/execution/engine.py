"""Reference event-timed spot execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ExecutionResult:
    """Trades and skipped-fill diagnostics for one symbol-day."""

    trades: pl.DataFrame
    skipped_entries: int
    skipped_exits: int


TRADE_SCHEMA = {
    "symbol": pl.String,
    "date": pl.String,
    "decision_time_ns": pl.Int64,
    "entry_time_ns": pl.Int64,
    "exit_time_ns": pl.Int64,
    "direction": pl.Int8,
    "prediction": pl.Float64,
    "signal_threshold": pl.Float64,
    "entry_price": pl.Float64,
    "exit_price": pl.Float64,
    "size": pl.Float64,
    "volatility": pl.Float64,
    "gross_return": pl.Float64,
    "gross_pnl": pl.Float64,
    "holding_seconds": pl.Float64,
}


def execute_non_overlapping(
    decisions: pl.DataFrame,
    spot_trades: pl.DataFrame,
    *,
    symbol: str,
    day: str,
    latency_ms: int,
    holding_seconds: int,
    threshold: float,
) -> ExecutionResult:
    """Execute threshold signals on first eligible future spot trades."""

    if latency_ms <= 0:
        raise ValueError("latency must be strictly positive")
    required_decisions = {
        "decision_time_ns",
        "prediction",
        "size",
        "spot_realised_volatility_5000ms",
    }
    if not required_decisions <= set(decisions.columns):
        raise ValueError("decision frame is missing execution columns")
    if not {"event_time_ns", "price"} <= set(spot_trades.columns):
        raise ValueError("spot trades are missing event_time_ns or price")

    event_times = spot_trades["event_time_ns"].to_numpy().astype(np.int64)
    prices = spot_trades["price"].to_numpy().astype(np.float64)
    if event_times.size and np.any(event_times[1:] < event_times[:-1]):
        raise ValueError("spot trades must be time ordered")

    rows: list[dict[str, Any]] = []
    skipped_entries = 0
    skipped_exits = 0
    active_until = -1
    latency_ns = latency_ms * 1_000_000
    holding_ns = holding_seconds * 1_000_000_000
    for row in decisions.sort("decision_time_ns").iter_rows(named=True):
        decision_time = int(row["decision_time_ns"])
        prediction = float(row["prediction"])
        if abs(prediction) < threshold or decision_time <= active_until:
            continue
        direction = 1 if prediction > 0 else -1
        entry_index = int(np.searchsorted(event_times, decision_time + latency_ns, side="left"))
        if entry_index >= event_times.size:
            skipped_entries += 1
            continue
        entry_time = int(event_times[entry_index])
        exit_index = int(np.searchsorted(event_times, entry_time + holding_ns, side="left"))
        if exit_index >= event_times.size:
            skipped_exits += 1
            continue
        exit_time = int(event_times[exit_index])
        entry_price = float(prices[entry_index])
        exit_price = float(prices[exit_index])
        size = float(row["size"])
        gross_return = direction * (exit_price / entry_price - 1.0)
        rows.append(
            {
                "symbol": symbol,
                "date": day,
                "decision_time_ns": decision_time,
                "entry_time_ns": entry_time,
                "exit_time_ns": exit_time,
                "direction": direction,
                "prediction": prediction,
                "signal_threshold": threshold,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": size,
                "volatility": float(row["spot_realised_volatility_5000ms"]),
                "gross_return": gross_return,
                "gross_pnl": size * gross_return,
                "holding_seconds": (exit_time - entry_time) / 1_000_000_000,
            }
        )
        active_until = exit_time
    trades = pl.DataFrame(rows, schema=TRADE_SCHEMA) if rows else pl.DataFrame(schema=TRADE_SCHEMA)
    return ExecutionResult(trades, skipped_entries, skipped_exits)


def apply_roundtrip_cost(trades: pl.DataFrame, roundtrip_cost_bps: float) -> pl.DataFrame:
    """Apply an additive all-in cost without changing gross P&L."""

    if roundtrip_cost_bps < 0:
        raise ValueError("cost cannot be negative")
    return trades.with_columns(
        (pl.col("size") * roundtrip_cost_bps / 10_000).alias("cost_pnl")
    ).with_columns((pl.col("gross_pnl") - pl.col("cost_pnl")).alias("net_pnl"))
