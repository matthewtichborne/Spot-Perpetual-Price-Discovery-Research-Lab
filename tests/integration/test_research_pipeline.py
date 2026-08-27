from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from spot_perp_lab.config import AppConfig
from spot_perp_lab.features.pipeline import feature_output_path
from spot_perp_lab.research.baselines import EXPANDED_FEATURES
from spot_perp_lab.research.pipeline import run_phase4


def test_phase4_pipeline_writes_complete_reports(tmp_path: Path) -> None:
    start = date(2025, 1, 2)
    generator = np.random.default_rng(12)
    feature_root = tmp_path / "features"
    for offset in range(20):
        day = start + timedelta(days=offset)
        rows = 40
        day_start_ns = int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 10**9
        values = {name: generator.normal(size=rows) for name in EXPANDED_FEATURES}
        target = 0.01 * values["perpetual_quantity_imbalance_1000ms"] + generator.normal(
            scale=0.1, size=rows
        )
        frame = pl.DataFrame(
            {
                "symbol": ["BTCUSDT"] * rows,
                "date": [day.isoformat()] * rows,
                "decision_time_ns": day_start_ns + 20_000_000_000 + np.arange(rows) * 5_000_000_000,
                **values,
                "target_spot_log_return_5000ms": target,
                "target_spot_direction_5000ms": (target > 0).astype(np.int8),
            }
        )
        path = feature_output_path(feature_root, "synthetic-development", "BTCUSDT", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)

    reports = tmp_path / "reports"
    config = AppConfig.model_validate(
        {
            "name": "synthetic-development",
            "data": {
                "symbols": ["BTCUSDT"],
                "dates": {
                    "start": start.isoformat(),
                    "end": (start + timedelta(days=19)).isoformat(),
                },
                "manifest_dir": tmp_path / "manifests",
            },
            "research": {
                "holdout_status": "not_applicable",
                "design_status": "frozen",
                "initial_train_days": 10,
                "test_days": 5,
                "purge_seconds": 10,
                "bootstrap_replicates": 100,
                "reports_dir": reports,
            },
            "features": {"output_dir": feature_root},
        }
    )
    result = run_phase4(config)
    assert result["folds"] == 2
    assert result["failures"] == 0
    assert (reports / "phase4_fold_metrics.csv").exists()
    assert (reports / "phase4_daily_metrics.csv").exists()
    assert (reports / "phase4_deciles.csv").exists()
    assert (reports / "phase4_bootstrap.csv").exists()
    assert (reports / "phase4_hac.csv").exists()
    assert (reports / "phase4_failures.csv").read_text().startswith("fold,task,model")
    assert (reports / "phase4_summary.md").exists()
