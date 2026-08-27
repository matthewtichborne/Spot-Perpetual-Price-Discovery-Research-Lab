"""Registered one-time Phase 8 final-holdout evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from spot_perp_lab.config import AppConfig
from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.checksums import sha256_file
from spot_perp_lab.data.manifest import canonical_json
from spot_perp_lab.data.normalise import processed_path
from spot_perp_lab.data.pipeline import SealedHoldoutError
from spot_perp_lab.execution.config import Phase6Config
from spot_perp_lab.execution.engine import apply_roundtrip_cost, execute_non_overlapping
from spot_perp_lab.execution.metrics import daily_pnl, performance_metrics
from spot_perp_lab.research.baselines import EXPANDED_FEATURES
from spot_perp_lab.research.evaluation import regression_metrics
from spot_perp_lab.research.models import xgboost_regressor
from spot_perp_lab.research.phase5 import PRIMARY_TARGET, _load_features

FINAL_START = "2025-02-21"
FINAL_END = "2025-03-02"


def _deciles(actual: np.ndarray, predicted: np.ndarray) -> pl.DataFrame:
    order = np.argsort(predicted, kind="stable")
    groups = np.empty(predicted.size, dtype=np.int8)
    for decile, indices in enumerate(np.array_split(order, 10), start=1):
        groups[indices] = decile
    return (
        pl.DataFrame({"decile": groups, "actual": actual, "predicted": predicted})
        .group_by("decile")
        .agg(
            pl.len().alias("rows"),
            pl.col("predicted").mean().alias("mean_prediction"),
            pl.col("actual").mean().alias("mean_actual"),
        )
        .sort("decile")
    )


def _daily_predictive(
    evaluation: pl.DataFrame, actual: np.ndarray, predicted: np.ndarray, training_mean: float
) -> pl.DataFrame:
    dates = evaluation["date"].to_numpy()
    rows = []
    for day in sorted(set(dates.tolist())):
        mask = dates == day
        rows.append(
            {"date": day, **regression_metrics(actual[mask], predicted[mask], training_mean)}
        )
    return pl.DataFrame(rows)


def _execute_final(
    final_config: AppConfig,
    signals: pl.DataFrame,
    threshold: float,
    phase6_config: Phase6Config,
) -> tuple[pl.DataFrame, int, int]:
    frames: list[pl.DataFrame] = []
    skipped_entries = 0
    skipped_exits = 0
    for day in sorted(signals["date"].unique().to_list()):
        spot = pl.read_parquet(
            processed_path(
                final_config.data.processed_dir,
                MarketType.SPOT,
                "BTCUSDT",
                datetime.strptime(day, "%Y-%m-%d").date(),
            ),
            columns=["event_time_ns", "price"],
            hive_partitioning=False,
        )
        result = execute_non_overlapping(
            signals.filter(pl.col("date") == day),
            spot,
            symbol="BTCUSDT",
            day=day,
            latency_ms=phase6_config.primary_latency_ms,
            holding_seconds=phase6_config.holding_seconds,
            threshold=threshold,
        )
        frames.append(result.trades)
        skipped_entries += result.skipped_entries
        skipped_exits += result.skipped_exits
    return pl.concat(frames), skipped_entries, skipped_exits


def _figure(
    daily: pl.DataFrame, deciles: pl.DataFrame, economic_daily: pl.DataFrame, path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(daily["date"], daily["oos_r2"], marker="o")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Daily predictive R²", xlabel="UTC date")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].plot(deciles["decile"], deciles["mean_actual"], marker="o")
    axes[1].set(title="Realised return by prediction decile", xlabel="Prediction decile")
    axes[2].plot(economic_daily["date"], economic_daily["gross_pnl"].cum_sum(), label="gross")
    axes[2].plot(economic_daily["date"], economic_daily["net_pnl"].cum_sum(), label="net")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set(title="Cumulative strategy P&L", xlabel="UTC date")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_phase8(
    development_config: AppConfig,
    confirmation_config: AppConfig,
    final_config: AppConfig,
    phase6_config: Phase6Config,
    *,
    final_specification_path: Path | None = None,
    phase6_manifest_path: Path | None = None,
    protocol_path: Path = Path("docs/phase8_protocol.md"),
    open_config_path: Path = Path("configs/final-open.yaml"),
) -> dict[str, Any]:
    """Fit the frozen model and evaluate the final BTC holdout exactly once."""

    manifest_path = final_config.data.manifest_dir / "final-evaluation.json"
    if manifest_path.exists():
        raise RuntimeError(f"final holdout has already been evaluated: {manifest_path}")
    if final_config.research.holdout_status != "open":
        raise SealedHoldoutError(f"configuration {final_config.name!r} is not open")
    if final_config.research.design_status != "frozen":
        raise ValueError("final evaluation requires a frozen design")
    actual_dates = (
        final_config.data.dates.start.isoformat(),
        final_config.data.dates.end.isoformat(),
    )
    if actual_dates != (FINAL_START, FINAL_END):
        raise ValueError("final evaluation dates differ from the registered holdout")

    specification_path = final_specification_path or (
        final_config.data.manifest_dir / "final-model-specification.json"
    )
    execution_path = phase6_manifest_path or (
        final_config.data.manifest_dir / "phase6-execution.json"
    )
    specification = json.loads(specification_path.read_text())
    execution = json.loads(execution_path.read_text())
    if specification["model"] != "xgboost" or specification["scope"] != "expanded":
        raise ValueError("frozen final model is not XGBoost/expanded")
    if tuple(specification["features"]) != EXPANDED_FEATURES:
        raise ValueError("frozen feature list differs from the implementation")
    if specification["target"] != PRIMARY_TARGET:
        raise ValueError("frozen target differs from the implementation")

    training = pl.concat(
        [
            _load_features(development_config, "BTCUSDT"),
            _load_features(confirmation_config, "BTCUSDT"),
        ]
    ).sort("decision_time_ns")
    eligible_training = training.filter(pl.col(PRIMARY_TARGET).is_not_null())
    sampled_training = eligible_training.filter(pl.col("decision_time_ns") % 5_000_000_000 == 0)
    evaluation = _load_features(final_config, "BTCUSDT").filter(
        pl.col(PRIMARY_TARGET).is_not_null()
    )
    parameters: dict[str, int | float] = {
        "max_depth": int(specification["parameters"]["max_depth"]),
        "n_estimators": int(specification["parameters"]["n_estimators"]),
    }
    estimator = xgboost_regressor(parameters, final_config.research.random_seed)
    estimator.fit(
        sampled_training.select(EXPANDED_FEATURES).to_numpy(),
        sampled_training[PRIMARY_TARGET].to_numpy(),
    )
    predicted = estimator.predict(evaluation.select(EXPANDED_FEATURES).to_numpy()).astype(
        np.float64
    )
    actual = evaluation[PRIMARY_TARGET].to_numpy().astype(np.float64)
    training_mean = float(np.mean(eligible_training[PRIMARY_TARGET].to_numpy()))
    predictive: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "target": PRIMARY_TARGET,
        "train_rows": sampled_training.height,
        "final_rows": evaluation.height,
        **regression_metrics(actual, predicted, training_mean),
    }
    daily = _daily_predictive(evaluation, actual, predicted, training_mean)
    deciles = _deciles(actual, predicted)

    threshold = float(execution["thresholds"]["BTCUSDT"])
    signals = evaluation.select(
        "date", "decision_time_ns", "spot_realised_volatility_5000ms"
    ).with_columns(pl.Series("prediction", predicted), pl.lit(1.0).alias("size"))
    gross_trades, skipped_entries, skipped_exits = _execute_final(
        final_config, signals, threshold, phase6_config
    )
    trades = apply_roundtrip_cost(gross_trades, phase6_config.reference_roundtrip_cost_bps)
    dates = tuple(
        (final_config.data.dates.start + timedelta(days=offset)).isoformat()
        for offset in range((final_config.data.dates.end - final_config.data.dates.start).days + 1)
    )
    economic_daily = daily_pnl(trades, dates)
    economic: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "latency_ms": phase6_config.primary_latency_ms,
        "holding_seconds": phase6_config.holding_seconds,
        "roundtrip_cost_bps": phase6_config.reference_roundtrip_cost_bps,
        "signal_threshold": threshold,
        "skipped_entries": skipped_entries,
        "skipped_exits": skipped_exits,
        **performance_metrics(trades, economic_daily, phase6_config.annualisation_days),
    }

    reports = final_config.research.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictive": reports / "final_predictive_metrics.csv",
        "daily_predictive": reports / "final_daily_metrics.csv",
        "deciles": reports / "final_deciles.csv",
        "economic": reports / "final_economic_metrics.csv",
        "daily_economic": reports / "final_economic_daily.csv",
        "summary": reports / "final_summary.md",
        "figure": reports / "final-results.png",
    }
    pl.DataFrame([predictive]).write_csv(outputs["predictive"])
    daily.write_csv(outputs["daily_predictive"])
    deciles.write_csv(outputs["deciles"])
    pl.DataFrame([economic]).write_csv(outputs["economic"])
    economic_daily.write_csv(outputs["daily_economic"])
    predictive_conclusion = "positive" if predictive["oos_r2"] > 0 else "non-positive"
    economic_conclusion = "positive" if economic["net_pnl"] > 0 else "non-positive"
    outputs["summary"].write_text(
        "# Final holdout result\n\n"
        f"The registered predictive result was **{predictive_conclusion}** and the registered "
        f"net economic result was **{economic_conclusion}**.\n\n"
        f"- Eligible final observations: {evaluation.height:,}\n"
        f"- Out-of-sample R-squared: {predictive['oos_r2']:.6g}\n"
        f"- Pearson / rank IC: {predictive['pearson_ic']:.6g} / {predictive['rank_ic']:.6g}\n"
        f"- Trades: {economic['trades']:,}\n"
        f"- Gross / net P&L: {economic['gross_pnl']:.6g} / {economic['net_pnl']:.6g}\n"
        f"- Break-even round-trip cost: {economic['break_even_roundtrip_bps']:.6g} bps\n\n"
        "This is the sole final evaluation under the frozen Phase 8 protocol. Aggregate trades "
        "do not reconstruct an executable order book, so economic results remain a sensitivity "
        "analysis rather than a live-trading claim.\n",
        encoding="utf-8",
    )
    _figure(daily, deciles, economic_daily, outputs["figure"])
    ledger_path = final_config.data.processed_dir / "final" / "final-primary-trades.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    trades.write_parquet(ledger_path)

    payload = {
        "phase": 8,
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "final_dates": {"start": FINAL_START, "end": FINAL_END},
        "model_specification_hash": specification["specification_hash"],
        "phase6_manifest_hash": execution["manifest_hash"],
        "protocol_sha256": sha256_file(protocol_path),
        "open_config_sha256": sha256_file(open_config_path),
        "training_rows": sampled_training.height,
        "final_rows": evaluation.height,
        "predictive": predictive,
        "economic": economic,
        "outputs": {name: sha256_file(path) for name, path in outputs.items()},
        "ledger_sha256": sha256_file(ledger_path),
        "one_time_evaluation": True,
    }
    manifest_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    manifest_path.write_text(
        json.dumps({**payload, "manifest_hash": manifest_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_hash": manifest_hash,
        "oos_r2": predictive["oos_r2"],
        "net_pnl": economic["net_pnl"],
        "trades": economic["trades"],
    }
