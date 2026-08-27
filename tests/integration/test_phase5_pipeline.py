from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from spot_perp_lab.config import AppConfig
from spot_perp_lab.features.pipeline import feature_output_path
from spot_perp_lab.research.baselines import EXPANDED_FEATURES
from spot_perp_lab.research.phase5 import run_phase5


def _write_features(
    root: Path, config_name: str, symbol: str, start: date, days: int, seed: int
) -> None:
    generator = np.random.default_rng(seed)
    for offset in range(days):
        day = start + timedelta(days=offset)
        rows = 24
        start_ns = int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 10**9
        values = {name: generator.normal(size=rows) for name in EXPANDED_FEATURES}
        signal = 0.002 * values["perpetual_quantity_imbalance_1000ms"]
        targets = {
            horizon: signal + generator.normal(scale=0.01 * horizon / 5_000, size=rows)
            for horizon in (1_000, 5_000, 10_000)
        }
        frame = pl.DataFrame(
            {
                "date": [day.isoformat()] * rows,
                "decision_time_ns": start_ns + 20_000_000_000 + np.arange(rows) * 5_000_000_000,
                **values,
                **{
                    f"target_spot_log_return_{horizon}ms": target
                    for horizon, target in targets.items()
                },
            }
        )
        path = feature_output_path(root, config_name, symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _config(
    name: str, start: date, end: date, feature_root: Path, manifest_root: Path, reports: Path
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "name": name,
            "data": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "dates": {"start": start.isoformat(), "end": end.isoformat()},
                "manifest_dir": manifest_root,
            },
            "research": {
                "holdout_status": "not_applicable",
                "design_status": "frozen",
                "bootstrap_replicates": 100,
                "reports_dir": reports,
            },
            "features": {"output_dir": feature_root},
        }
    )


def test_phase5_pipeline_writes_confirmation_and_final_specification(tmp_path: Path) -> None:
    feature_root = tmp_path / "features"
    manifests = tmp_path / "manifests"
    reports = tmp_path / "reports"
    development_start = date(2025, 1, 2)
    confirmation_start = date(2025, 2, 1)
    for symbol, seed in (("BTCUSDT", 3), ("ETHUSDT", 7)):
        _write_features(feature_root, "synthetic-development", symbol, development_start, 30, seed)
        _write_features(
            feature_root, "synthetic-confirmation", symbol, confirmation_start, 20, seed + 1
        )
    manifests.mkdir()
    for name in ("synthetic-development", "synthetic-confirmation"):
        (manifests / f"{name}-features.json").write_text(
            json.dumps({"manifest_hash": f"{name}-hash"}), encoding="utf-8"
        )
    development = _config(
        "synthetic-development",
        development_start,
        date(2025, 1, 31),
        feature_root,
        manifests,
        reports,
    )
    confirmation = _config(
        "synthetic-confirmation",
        confirmation_start,
        date(2025, 2, 20),
        feature_root,
        manifests,
        reports,
    )
    result = run_phase5(development, confirmation)
    assert result["selected_model"] in {"ridge", "xgboost"}
    assert result["failures"] == 0
    assert (reports / "phase5_model_metrics.csv").exists()
    assert (reports / "phase5_xgboost_tuning.csv").exists()
    assert (reports / "phase5_regimes.csv").exists()
    assert (reports / "phase5_horizons.csv").exists()
    assert (reports / "phase5_eth_replication.csv").exists()
    assert (reports / "phase5_placebo.csv").exists()
    assert (reports / "phase5_summary.md").exists()
    final_spec = json.loads((manifests / "final-model-specification.json").read_text())
    assert final_spec["final_holdout_status"] == "sealed"
    assert final_spec["target"] == "target_spot_log_return_5000ms"
