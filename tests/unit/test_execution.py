from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from spot_perp_lab.execution.config import Phase6Config
from spot_perp_lab.execution.engine import apply_roundtrip_cost, execute_non_overlapping
from spot_perp_lab.execution.metrics import daily_pnl, portfolio_ledger


def test_first_eligible_fill_is_after_latency_and_positions_do_not_overlap() -> None:
    origin = 1_000_000_000_000
    decisions = pl.DataFrame(
        {
            "decision_time_ns": [origin, origin + 1_000_000_000, origin + 6_000_000_000],
            "prediction": [2.0, 2.0, -2.0],
            "size": [1.0, 1.0, 1.0],
            "spot_realised_volatility_5000ms": [0.1, 0.1, 0.1],
        }
    )
    spot = pl.DataFrame(
        {
            "event_time_ns": [
                origin,
                origin + 100_000_000,
                origin + 5_100_000_000,
                origin + 6_100_000_000,
                origin + 11_100_000_000,
            ],
            "price": [100.0, 101.0, 102.0, 103.0, 102.0],
        }
    )
    result = execute_non_overlapping(
        decisions,
        spot,
        symbol="BTCUSDT",
        day="2025-02-01",
        latency_ms=100,
        holding_seconds=5,
        threshold=1.0,
    )
    assert result.trades.height == 2
    assert result.trades["entry_time_ns"].to_list() == [
        origin + 100_000_000,
        origin + 6_100_000_000,
    ]
    assert all(
        entry > decision
        for entry, decision in zip(
            result.trades["entry_time_ns"],
            result.trades["decision_time_ns"],
            strict=True,
        )
    )
    assert (
        result.trades["decision_time_ns"].to_list()[1] > result.trades["exit_time_ns"].to_list()[0]
    )


def test_costs_are_separate_and_daily_pnl_reconciles() -> None:
    trades = pl.DataFrame(
        {
            "date": ["2025-02-01", "2025-02-02"],
            "size": [1.0, 0.5],
            "gross_pnl": [0.001, -0.0002],
            "holding_seconds": [5.0, 5.0],
        }
    )
    costed = apply_roundtrip_cost(trades, 7.0)
    assert costed["cost_pnl"].to_list() == pytest.approx([0.0007, 0.00035])
    assert costed["net_pnl"].to_list() == pytest.approx([0.0003, -0.00055])
    daily = daily_pnl(costed, ("2025-02-01", "2025-02-02", "2025-02-03"))
    assert daily.height == 3
    assert float(daily["gross_pnl"].sum()) == pytest.approx(float(costed["gross_pnl"].sum()))
    assert float(daily["net_pnl"].sum()) == pytest.approx(float(costed["net_pnl"].sum()))


def test_portfolio_stops_new_entries_after_realised_daily_loss_limit() -> None:
    base = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "date": ["2025-02-01", "2025-02-01"],
            "decision_time_ns": [0, 20],
            "entry_time_ns": [1, 21],
            "exit_time_ns": [10, 30],
            "direction": [1, 1],
            "prediction": [1.0, 1.0],
            "signal_threshold": [0.5, 0.5],
            "entry_price": [100.0, 100.0],
            "exit_price": [97.0, 101.0],
            "size": [1.0, 1.0],
            "volatility": [0.1, 0.1],
            "gross_return": [-0.03, 0.01],
            "gross_pnl": [-0.03, 0.01],
            "holding_seconds": [5.0, 5.0],
        }
    )
    portfolio = portfolio_ledger({"BTCUSDT": base}, {"BTCUSDT": 1.0}, 0.0, 0.02)
    assert portfolio.height == 1
    assert portfolio["gross_pnl"].to_list() == [-0.03]


def test_phase6_config_requires_reference_cost_in_grid() -> None:
    payload = {
        "name": "test",
        "development_config": "development.yaml",
        "evaluation_config": "confirmation.yaml",
        "reports_dir": "reports",
        "ledger_dir": "ledger",
        "holding_seconds": 5,
        "signal_threshold_sigma": 1.0,
        "latencies_ms": [100, 500],
        "primary_latency_ms": 500,
        "entry_fee_bps": 2.0,
        "exit_fee_bps": 2.0,
        "slippage_bps_per_side": 1.0,
        "spread_proxy_bps_per_side": 0.5,
        "cost_sensitivity_roundtrip_bps": [0.0, 7.0],
        "maximum_gross_exposure": 1.0,
        "inverse_volatility_minimum_multiplier": 0.25,
        "inverse_volatility_maximum_multiplier": 2.0,
        "daily_loss_limit": 0.02,
        "annualisation_days": 365,
        "random_seed": 7,
    }
    assert Phase6Config.model_validate(payload).reference_roundtrip_cost_bps == 7.0
    payload["cost_sensitivity_roundtrip_bps"] = [0.0, 5.0]
    with pytest.raises(ValueError, match="reference all-in cost"):
        Phase6Config.model_validate(payload)


def test_execution_rejects_unsorted_spot_events() -> None:
    decisions = pl.DataFrame(
        {
            "decision_time_ns": [0],
            "prediction": [1.0],
            "size": [1.0],
            "spot_realised_volatility_5000ms": [0.1],
        }
    )
    spot = pl.DataFrame({"event_time_ns": np.array([2, 1]), "price": [1.0, 1.0]})
    with pytest.raises(ValueError, match="time ordered"):
        execute_non_overlapping(
            decisions,
            spot,
            symbol="BTCUSDT",
            day="2025-02-01",
            latency_ms=100,
            holding_seconds=5,
            threshold=0.5,
        )
