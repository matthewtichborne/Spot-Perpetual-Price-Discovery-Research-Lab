from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from spot_perp_lab.config import AppConfig
from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.normalise import processed_path
from spot_perp_lab.execution.config import Phase6Config
from spot_perp_lab.features.pipeline import feature_output_path
from spot_perp_lab.research.baselines import EXPANDED_FEATURES
from spot_perp_lab.research.phase8 import run_phase8


def _app(root: Path, name: str, start: date, end: date, status: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "name": name,
            "data": {
                "symbols": ["BTCUSDT"],
                "dates": {"start": start, "end": end},
                "processed_dir": root / "processed",
                "manifest_dir": root / "manifests",
            },
            "research": {
                "holdout_status": status,
                "design_status": "frozen",
                "reports_dir": root / "reports" / "final",
                "random_seed": 7,
            },
            "features": {"output_dir": root / "features"},
        }
    )


def _features(root: Path, name: str, start: date, days: int, seed: int) -> None:
    generator = np.random.default_rng(seed)
    for offset in range(days):
        day = start + timedelta(days=offset)
        rows = 24
        start_ns = int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 10**9
        values = {feature: generator.normal(size=rows) for feature in EXPANDED_FEATURES}
        target = 0.002 * values["perpetual_quantity_imbalance_1000ms"]
        frame = pl.DataFrame(
            {
                "date": [day.isoformat()] * rows,
                "decision_time_ns": start_ns + 20_000_000_000 + np.arange(rows) * 5_000_000_000,
                **values,
                "target_spot_log_return_1000ms": target,
                "target_spot_log_return_5000ms": target + generator.normal(0, 0.001, rows),
                "target_spot_log_return_10000ms": target,
            }
        )
        path = feature_output_path(root / "features", name, "BTCUSDT", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _spot(root: Path, start: date, days: int) -> None:
    for offset in range(days):
        day = start + timedelta(days=offset)
        start_ns = int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 10**9
        times = start_ns + np.arange(180) * 1_000_000_000
        path = processed_path(root / "processed", MarketType.SPOT, "BTCUSDT", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"event_time_ns": times, "price": 100 + np.arange(180) * 0.001}).write_parquet(
            path
        )


def test_phase8_runs_once_and_enforces_future_fills(tmp_path: Path) -> None:
    development = _app(
        tmp_path, "synthetic-development", date(2025, 1, 2), date(2025, 1, 31), "not_applicable"
    )
    confirmation = _app(
        tmp_path, "synthetic-confirmation", date(2025, 2, 1), date(2025, 2, 20), "not_applicable"
    )
    final = _app(tmp_path, "synthetic-final", date(2025, 2, 21), date(2025, 3, 2), "open")
    _features(tmp_path, development.name, date(2025, 1, 2), 30, 1)
    _features(tmp_path, confirmation.name, date(2025, 2, 1), 20, 2)
    _features(tmp_path, final.name, date(2025, 2, 21), 10, 3)
    _spot(tmp_path, date(2025, 2, 21), 10)
    manifests = tmp_path / "manifests"
    manifests.mkdir(exist_ok=True)
    specification = manifests / "spec.json"
    specification.write_text(
        json.dumps(
            {
                "model": "xgboost",
                "scope": "expanded",
                "target": "target_spot_log_return_5000ms",
                "features": list(EXPANDED_FEATURES),
                "parameters": {"max_depth": 2, "n_estimators": 20},
                "specification_hash": "synthetic-specification",
            }
        ),
        encoding="utf-8",
    )
    execution = manifests / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "thresholds": {"BTCUSDT": 0.00001},
                "manifest_hash": "synthetic-execution",
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.md"
    open_config = tmp_path / "final-open.yaml"
    protocol.write_text("frozen", encoding="utf-8")
    open_config.write_text("open", encoding="utf-8")
    phase6 = Phase6Config.model_validate(
        {
            "name": "synthetic-phase6",
            "development_config": tmp_path / "development.yaml",
            "evaluation_config": tmp_path / "confirmation.yaml",
            "reports_dir": tmp_path / "reports",
            "ledger_dir": tmp_path / "ledger",
            "holding_seconds": 5,
            "signal_threshold_sigma": 1,
            "latencies_ms": [500],
            "primary_latency_ms": 500,
            "entry_fee_bps": 2,
            "exit_fee_bps": 2,
            "slippage_bps_per_side": 1,
            "spread_proxy_bps_per_side": 0.5,
            "cost_sensitivity_roundtrip_bps": [7],
            "maximum_gross_exposure": 1,
            "inverse_volatility_minimum_multiplier": 0.25,
            "inverse_volatility_maximum_multiplier": 2,
            "daily_loss_limit": 0.02,
            "annualisation_days": 365,
            "random_seed": 7,
        }
    )
    result = run_phase8(
        development,
        confirmation,
        final,
        phase6,
        final_specification_path=specification,
        phase6_manifest_path=execution,
        protocol_path=protocol,
        open_config_path=open_config,
    )
    assert np.isfinite(result["oos_r2"])
    ledger = pl.read_parquet(tmp_path / "processed" / "final" / "final-primary-trades.parquet")
    assert (ledger["entry_time_ns"] > ledger["decision_time_ns"]).all()
    assert (manifests / "final-evaluation.json").exists()
    try:
        run_phase8(development, confirmation, final, phase6)
    except RuntimeError as error:
        assert "already been evaluated" in str(error)
    else:
        raise AssertionError("second final evaluation was not refused")
