from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.normalise import processed_path
from spot_perp_lab.execution.config import Phase6Config
from spot_perp_lab.execution.pipeline import run_phase6
from spot_perp_lab.features.pipeline import feature_output_path
from spot_perp_lab.research.baselines import EXPANDED_FEATURES


def _write_feature_days(
    root: Path, name: str, symbol: str, start: date, days: int, seed: int
) -> None:
    generator = np.random.default_rng(seed)
    for offset in range(days):
        day = start + timedelta(days=offset)
        rows = 24
        start_ns = int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 10**9
        values = {feature: generator.normal(size=rows) for feature in EXPANDED_FEATURES}
        signal = 0.003 * values["perpetual_quantity_imbalance_1000ms"]
        frame = pl.DataFrame(
            {
                "date": [day.isoformat()] * rows,
                "decision_time_ns": start_ns + 20_000_000_000 + np.arange(rows) * 5_000_000_000,
                **values,
                "target_spot_log_return_1000ms": signal + generator.normal(scale=0.004, size=rows),
                "target_spot_log_return_5000ms": signal + generator.normal(scale=0.01, size=rows),
                "target_spot_log_return_10000ms": signal + generator.normal(scale=0.02, size=rows),
            }
        )
        path = feature_output_path(root, name, symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _write_spot_days(root: Path, symbol: str, start: date, days: int, seed: int) -> None:
    generator = np.random.default_rng(seed)
    for offset in range(days):
        day = start + timedelta(days=offset)
        start_ns = int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 10**9
        rows = 180
        returns = generator.normal(loc=0.00001, scale=0.0001, size=rows)
        prices = 100 * np.exp(np.cumsum(returns))
        frame = pl.DataFrame(
            {
                "event_time_ns": start_ns + np.arange(rows) * 1_000_000_000,
                "price": prices,
            }
        )
        path = processed_path(root, MarketType.SPOT, symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _write_app_config(
    path: Path,
    name: str,
    start: date,
    end: date,
    feature_root: Path,
    processed_root: Path,
    manifest_root: Path,
) -> None:
    payload = {
        "name": name,
        "data": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "dates": {"start": start.isoformat(), "end": end.isoformat()},
            "processed_dir": str(processed_root),
            "manifest_dir": str(manifest_root),
        },
        "research": {
            "holdout_status": "not_applicable",
            "design_status": "frozen",
        },
        "features": {"output_dir": str(feature_root)},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_phase6_pipeline_reconciles_and_never_uses_decision_event(tmp_path: Path) -> None:
    feature_root = tmp_path / "features"
    processed_root = tmp_path / "processed"
    manifest_root = tmp_path / "manifests"
    reports = tmp_path / "reports" / "development"
    development_start = date(2025, 1, 2)
    evaluation_start = date(2025, 2, 1)
    for symbol, seed in (("BTCUSDT", 10), ("ETHUSDT", 20)):
        _write_feature_days(
            feature_root, "synthetic-development", symbol, development_start, 30, seed
        )
        _write_feature_days(
            feature_root, "synthetic-confirmation", symbol, evaluation_start, 20, seed + 1
        )
        _write_spot_days(processed_root, symbol, evaluation_start, 20, seed + 2)
    manifest_root.mkdir()
    (manifest_root / "final-model-specification.json").write_text(
        json.dumps(
            {
                "model": "xgboost",
                "scope": "expanded",
                "specification_hash": "synthetic-specification",
                "parameters": {"max_depth": 2, "n_estimators": 100},
            }
        ),
        encoding="utf-8",
    )
    development_config = tmp_path / "development.yaml"
    evaluation_config = tmp_path / "confirmation.yaml"
    _write_app_config(
        development_config,
        "synthetic-development",
        development_start,
        date(2025, 1, 31),
        feature_root,
        processed_root,
        manifest_root,
    )
    _write_app_config(
        evaluation_config,
        "synthetic-confirmation",
        evaluation_start,
        date(2025, 2, 20),
        feature_root,
        processed_root,
        manifest_root,
    )
    config = Phase6Config.model_validate(
        {
            "name": "synthetic-phase6",
            "development_config": development_config,
            "evaluation_config": evaluation_config,
            "reports_dir": reports,
            "ledger_dir": tmp_path / "ledger",
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
    )
    result = run_phase6(config)
    assert result["primary_trades"] >= 0
    asset_ledger = pl.read_parquet(tmp_path / "ledger" / "phase6-primary-asset-trades.parquet")
    assert (asset_ledger["entry_time_ns"] > asset_ledger["decision_time_ns"]).all()
    daily = pl.read_csv(reports / "phase6_daily.csv")
    for entity in ("BTCUSDT", "ETHUSDT", "combined"):
        reported = float(daily.filter(pl.col("entity") == entity)["net_pnl"].sum())
        if entity == "combined":
            ledger = pl.read_parquet(
                tmp_path / "ledger" / "phase6-primary-portfolio-trades.parquet"
            )
        else:
            ledger = asset_ledger.filter(pl.col("symbol") == entity)
        assert np.isclose(reported, float(ledger["net_pnl"].sum()), rtol=0, atol=1e-12)
    assert (reports / "phase6_summary.md").exists()
    assert (manifest_root / "phase6-execution.json").exists()
