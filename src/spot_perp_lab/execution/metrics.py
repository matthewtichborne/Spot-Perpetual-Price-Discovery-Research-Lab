"""Trade, daily, portfolio, and risk metrics for Phase 6."""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np
import polars as pl


def daily_pnl(trades: pl.DataFrame, dates: tuple[str, ...]) -> pl.DataFrame:
    """Aggregate closed-trade P&L and retain zero-trade UTC days."""

    if trades.is_empty():
        return pl.DataFrame({"date": dates}).with_columns(
            pl.lit(0).cast(pl.Int64).alias("trades"),
            pl.lit(0.0).alias("gross_pnl"),
            pl.lit(0.0).alias("cost_pnl"),
            pl.lit(0.0).alias("net_pnl"),
        )
    trade_dates = trades["date"].to_numpy()
    gross = trades["gross_pnl"].to_numpy().astype(np.float64)
    cost = trades["cost_pnl"].to_numpy().astype(np.float64)
    net = trades["net_pnl"].to_numpy().astype(np.float64)
    rows = []
    for day in dates:
        mask = trade_dates == day
        rows.append(
            {
                "date": day,
                "trades": int(np.sum(mask)),
                "gross_pnl": float(np.sum(gross[mask])),
                "cost_pnl": float(np.sum(cost[mask])),
                "net_pnl": float(np.sum(net[mask])),
            }
        )
    return pl.DataFrame(rows)


def performance_metrics(
    trades: pl.DataFrame, daily: pl.DataFrame, annualisation_days: int
) -> dict[str, float | int]:
    """Compute non-annualised trade and annualised daily performance statistics."""

    net_daily = daily["net_pnl"].to_numpy().astype(np.float64)
    mean_daily = float(np.mean(net_daily)) if net_daily.size else 0.0
    daily_std = float(np.std(net_daily, ddof=1)) if net_daily.size > 1 else 0.0
    downside = net_daily[net_daily < 0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sharpe = np.sqrt(annualisation_days) * mean_daily / daily_std if daily_std > 0 else 0.0
    sortino = np.sqrt(annualisation_days) * mean_daily / downside_std if downside_std > 0 else 0.0
    equity = 1.0 + np.cumsum(net_daily)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = np.maximum(equity, 0.0) / peaks - 1.0 if equity.size else np.array([0.0])
    gross = trades["gross_pnl"].to_numpy() if not trades.is_empty() else np.array([])
    costs = trades["cost_pnl"].to_numpy() if not trades.is_empty() else np.array([])
    net = trades["net_pnl"].to_numpy() if not trades.is_empty() else np.array([])
    sizes = trades["size"].to_numpy() if not trades.is_empty() else np.array([])
    wins = net[net > 0]
    losses = net[net < 0]
    absolute_daily = np.abs(net_daily)
    concentration = (
        float(np.max(absolute_daily) / np.sum(absolute_daily))
        if np.sum(absolute_daily) > 0
        else 0.0
    )
    total_size = float(np.sum(sizes))
    exposure_seconds = (
        float(
            np.sum(
                trades["size"].to_numpy().astype(np.float64)
                * trades["holding_seconds"].to_numpy().astype(np.float64)
            )
        )
        if not trades.is_empty()
        else 0.0
    )
    return {
        "trades": trades.height,
        "gross_pnl": float(np.sum(gross)),
        "cost_pnl": float(np.sum(costs)),
        "net_pnl": float(np.sum(net)),
        "exposure": exposure_seconds / (max(daily.height, 1) * 86_400),
        "turnover": 2.0 * total_size,
        "annualised_daily_sharpe": float(sharpe),
        "annualised_daily_sortino": float(sortino),
        "maximum_drawdown": float(np.min(drawdown)),
        "win_rate": float(np.mean(net > 0)) if net.size else 0.0,
        "average_win": float(np.mean(wins)) if wins.size else 0.0,
        "average_loss": float(np.mean(losses)) if losses.size else 0.0,
        "average_holding_seconds": (
            float(np.mean(trades["holding_seconds"].to_numpy().astype(np.float64)))
            if not trades.is_empty()
            else 0.0
        ),
        "break_even_roundtrip_bps": (
            float(np.sum(gross) / total_size * 10_000) if total_size > 0 else 0.0
        ),
        "pnl_concentration_largest_day": concentration,
    }


def portfolio_ledger(
    asset_trades: dict[str, pl.DataFrame],
    weights: dict[str, float],
    roundtrip_cost_bps: float,
    daily_loss_limit: float,
) -> pl.DataFrame:
    """Weight asset trades and stop accepting entries after the daily loss limit."""

    frames = [
        frame.with_columns(
            (pl.col("size") * weights[symbol]).alias("size"),
            (pl.col("gross_pnl") * weights[symbol]).alias("gross_pnl"),
        )
        for symbol, frame in asset_trades.items()
        if not frame.is_empty()
    ]
    if not frames:
        return pl.DataFrame()
    candidates = (
        pl.concat(frames)
        .sort(["date", "entry_time_ns", "symbol"])
        .with_columns((pl.col("size") * roundtrip_cost_bps / 10_000).alias("cost_pnl"))
        .with_columns((pl.col("gross_pnl") - pl.col("cost_pnl")).alias("net_pnl"))
    )
    accepted: list[dict[str, Any]] = []
    current_day = ""
    realised = 0.0
    pending: list[tuple[int, int, float]] = []
    counter = 0
    for row in candidates.iter_rows(named=True):
        day = str(row["date"])
        if day != current_day:
            current_day = day
            realised = 0.0
            pending = []
        entry_time = int(row["entry_time_ns"])
        while pending and pending[0][0] <= entry_time:
            realised += heapq.heappop(pending)[2]
        if realised <= -daily_loss_limit:
            continue
        accepted.append(row)
        heapq.heappush(pending, (int(row["exit_time_ns"]), counter, float(row["net_pnl"])))
        counter += 1
    return pl.DataFrame(accepted, schema=candidates.schema)
