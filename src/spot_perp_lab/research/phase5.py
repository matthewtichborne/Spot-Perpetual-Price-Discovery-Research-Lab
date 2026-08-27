"""Frozen Phase 5 nonlinear comparison and robustness pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from spot_perp_lab.config import AppConfig
from spot_perp_lab.data.checksums import sha256_file
from spot_perp_lab.data.manifest import canonical_json
from spot_perp_lab.data.pipeline import SealedHoldoutError
from spot_perp_lab.features.pipeline import feature_output_path
from spot_perp_lab.research.baselines import BASELINE_FEATURES, EXPANDED_FEATURES
from spot_perp_lab.research.evaluation import regression_metrics
from spot_perp_lab.research.inference import paired_day_bootstrap
from spot_perp_lab.research.models import regression_pipeline, xgboost_regressor

HORIZONS_MS = (1_000, 5_000, 10_000)
PRIMARY_TARGET = "target_spot_log_return_5000ms"
XGBOOST_GRID: tuple[dict[str, int | float], ...] = (
    {"max_depth": 2, "n_estimators": 100},
    {"max_depth": 3, "n_estimators": 100},
    {"max_depth": 2, "n_estimators": 200},
    {"max_depth": 3, "n_estimators": 200},
)
XGBOOST_FIXED = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "training_subsample_seconds": 5,
}
PLACEBO_SHIFT_ROWS = 900


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _load_features(config: AppConfig, symbol: str) -> pl.DataFrame:
    paths = [
        feature_output_path(config.features.output_dir, config.name, symbol, day)
        for day in _days(config.data.dates.start, config.data.dates.end)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} feature partitions; first: {missing[0]}")
    targets = [f"target_spot_log_return_{horizon}ms" for horizon in HORIZONS_MS]
    columns = ["date", "decision_time_ns", *EXPANDED_FEATURES, *targets]
    return pl.concat(
        [pl.read_parquet(path, columns=columns, hive_partitioning=False) for path in paths]
    ).sort("decision_time_ns")


def _eligible(frame: pl.DataFrame, target: str) -> pl.DataFrame:
    return frame.filter(pl.col(target).is_not_null())


def _fit_ridge(
    development: pl.DataFrame,
    confirmation: pl.DataFrame,
    features: tuple[str, ...],
    target: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    train = _eligible(development, target)
    evaluation = _eligible(confirmation, target)
    y_train = train[target].to_numpy().astype(np.float64)
    y_eval = evaluation[target].to_numpy().astype(np.float64)
    estimator = regression_pipeline("ridge", alpha)
    estimator.fit(train.select(features).to_numpy(), y_train)
    prediction = estimator.predict(evaluation.select(features).to_numpy()).astype(np.float64)
    return y_eval, prediction, float(np.mean(y_train)), train.height


def _daily_metrics(
    frame: pl.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    training_mean: float,
    model: str,
    scope: str,
) -> list[dict[str, Any]]:
    dates = _eligible(frame, PRIMARY_TARGET)["date"].to_numpy()
    rows: list[dict[str, Any]] = []
    for day in sorted(set(dates.tolist())):
        mask = dates == day
        rows.append(
            {
                "date": day,
                "model": model,
                "scope": scope,
                **regression_metrics(actual[mask], predicted[mask], training_mean),
            }
        )
    return rows


def _metric_row(
    symbol: str,
    horizon_ms: int,
    model: str,
    scope: str,
    train_rows: int,
    actual: np.ndarray,
    predicted: np.ndarray,
    training_mean: float,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "horizon_ms": horizon_ms,
        "model": model,
        "scope": scope,
        "train_rows": train_rows,
        "confirmation_rows": actual.size,
        **regression_metrics(actual, predicted, training_mean),
    }


def _bootstrap_row(
    comparison: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left_mse = {row["date"]: row["mse"] for row in left}
    right_mse = {row["date"]: row["mse"] for row in right}
    dates = sorted(left_mse.keys() & right_mse.keys())
    improvements = np.array([left_mse[day] - right_mse[day] for day in dates])
    interval = paired_day_bootstrap(improvements, replicates, seed)
    return {"comparison": comparison, "days": len(dates), **asdict(interval)}


def _xgboost_tuning_masks(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    validation_start = datetime(2025, 1, 27, tzinfo=UTC)
    boundary_ns = int(validation_start.timestamp()) * 1_000_000_000
    train = frame.filter(
        (pl.col("decision_time_ns") < boundary_ns - 10_000_000_000)
        & (pl.col("decision_time_ns") % 5_000_000_000 == 0)
        & pl.col(PRIMARY_TARGET).is_not_null()
    )
    validation = frame.filter(
        (pl.col("decision_time_ns") >= boundary_ns + 10_000_000_000)
        & pl.col(PRIMARY_TARGET).is_not_null()
    )
    if not train.height or not validation.height:
        raise ValueError("Phase 5 XGBoost tuning split is empty")
    return train, validation


def _tune_xgboost(
    development: pl.DataFrame, seed: int, failures: list[dict[str, Any]]
) -> tuple[dict[str, int | float], list[dict[str, Any]]]:
    train, validation = _xgboost_tuning_masks(development)
    x_train = train.select(EXPANDED_FEATURES).to_numpy()
    y_train = train[PRIMARY_TARGET].to_numpy().astype(np.float64)
    x_validation = validation.select(EXPANDED_FEATURES).to_numpy()
    y_validation = validation[PRIMARY_TARGET].to_numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for parameters in XGBOOST_GRID:
        try:
            estimator = xgboost_regressor(parameters, seed)
            estimator.fit(x_train, y_train)
            prediction = estimator.predict(x_validation).astype(np.float64)
            rows.append(
                {
                    **parameters,
                    "train_rows": train.height,
                    "validation_rows": validation.height,
                    "validation_mse": float(np.mean(np.square(y_validation - prediction))),
                }
            )
        except Exception as error:
            failures.append(
                {
                    "check": f"xgboost_tuning_{parameters}",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    if not rows:
        raise RuntimeError("every registered XGBoost candidate failed")
    winner = min(
        rows,
        key=lambda row: (row["validation_mse"], row["n_estimators"], row["max_depth"]),
    )
    selected: dict[str, int | float] = {
        "max_depth": int(winner["max_depth"]),
        "n_estimators": int(winner["n_estimators"]),
    }
    return selected, rows


def _fit_xgboost(
    development: pl.DataFrame,
    confirmation: pl.DataFrame,
    parameters: dict[str, int | float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    train = _eligible(development, PRIMARY_TARGET).filter(
        pl.col("decision_time_ns") % 5_000_000_000 == 0
    )
    evaluation = _eligible(confirmation, PRIMARY_TARGET)
    y_train = train[PRIMARY_TARGET].to_numpy().astype(np.float64)
    y_eval = evaluation[PRIMARY_TARGET].to_numpy().astype(np.float64)
    estimator = xgboost_regressor(parameters, seed)
    estimator.fit(train.select(EXPANDED_FEATURES).to_numpy(), y_train)
    prediction = estimator.predict(evaluation.select(EXPANDED_FEATURES).to_numpy()).astype(
        np.float64
    )
    complete_training_mean = float(
        np.mean(_eligible(development, PRIMARY_TARGET)[PRIMARY_TARGET].to_numpy())
    )
    return y_eval, prediction, complete_training_mean, train.height


def _circular_placebo_matrix(frame: pl.DataFrame) -> np.ndarray:
    baseline = frame.select(BASELINE_FEATURES).to_numpy()
    additions = tuple(name for name in EXPANDED_FEATURES if name not in BASELINE_FEATURES)
    shifted = frame.select(additions).to_numpy()
    dates = frame["date"].to_numpy()
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    ends = np.r_[starts[1:], dates.size]
    for start, end in zip(starts, ends, strict=True):
        shifted[start:end] = np.roll(shifted[start:end], PLACEBO_SHIFT_ROWS, axis=0)
    return np.column_stack((baseline, shifted))


def _placebo_result(
    development: pl.DataFrame, confirmation: pl.DataFrame, alpha: float
) -> dict[str, Any]:
    train = _eligible(development, PRIMARY_TARGET)
    evaluation = _eligible(confirmation, PRIMARY_TARGET)
    y_train = train[PRIMARY_TARGET].to_numpy().astype(np.float64)
    y_eval = evaluation[PRIMARY_TARGET].to_numpy().astype(np.float64)
    estimator = regression_pipeline("ridge", alpha)
    estimator.fit(_circular_placebo_matrix(train), y_train)
    prediction = estimator.predict(_circular_placebo_matrix(evaluation)).astype(np.float64)
    return {
        "symbol": "BTCUSDT",
        "shift_rows": PLACEBO_SHIFT_ROWS,
        "train_rows": train.height,
        "confirmation_rows": evaluation.height,
        **regression_metrics(y_eval, prediction, float(np.mean(y_train))),
    }


def _regime_rows(
    development: pl.DataFrame,
    confirmation: pl.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    training_mean: float,
) -> list[dict[str, Any]]:
    threshold = float(np.median(development["spot_realised_volatility_5000ms"].to_numpy()))
    volatility = _eligible(confirmation, PRIMARY_TARGET)[
        "spot_realised_volatility_5000ms"
    ].to_numpy()
    rows: list[dict[str, Any]] = []
    for regime, mask in (("low", volatility < threshold), ("high", volatility >= threshold)):
        rows.append(
            {
                "regime": regime,
                "development_threshold": threshold,
                "confirmation_rows": int(np.sum(mask)),
                **regression_metrics(actual[mask], predicted[mask], training_mean),
            }
        )
    return rows


def _write_csv(
    path: Path, rows: list[dict[str, Any]], schema: dict[str, Any] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema or {})
    frame.write_csv(path)


def run_phase5(development_config: AppConfig, confirmation_config: AppConfig) -> dict[str, Any]:
    """Run the frozen Phase 5 confirmation and robustness procedure."""

    if development_config.research.holdout_status == "sealed":
        raise SealedHoldoutError(f"configuration {development_config.name!r} is sealed")
    if confirmation_config.research.holdout_status == "sealed":
        raise SealedHoldoutError(f"configuration {confirmation_config.name!r} is sealed")
    if (
        development_config.research.design_status != "frozen"
        or confirmation_config.research.design_status != "frozen"
    ):
        raise ValueError("Phase 5 requires frozen development and confirmation designs")
    if development_config.data.dates.end >= confirmation_config.data.dates.start:
        raise ValueError("development must end before confirmation begins")

    reports = confirmation_config.research.reports_dir
    model_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    eth_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    development = _load_features(development_config, "BTCUSDT")
    confirmation = _load_features(confirmation_config, "BTCUSDT")
    primary_predictions: dict[str, tuple[np.ndarray, np.ndarray, float, int]] = {}
    for scope, features in (("baseline", BASELINE_FEATURES), ("expanded", EXPANDED_FEATURES)):
        result = _fit_ridge(
            development,
            confirmation,
            features,
            PRIMARY_TARGET,
            development_config.research.ridge_alpha,
        )
        primary_predictions[f"ridge_{scope}"] = result
        actual, predicted, training_mean, train_rows = result
        model_rows.append(
            _metric_row(
                "BTCUSDT", 5_000, "ridge", scope, train_rows, actual, predicted, training_mean
            )
        )
        daily_rows.extend(
            _daily_metrics(confirmation, actual, predicted, training_mean, "ridge", scope)
        )

    selected_xgboost, tuning_rows = _tune_xgboost(
        development, development_config.research.random_seed, failures
    )
    xgb_result = _fit_xgboost(
        development,
        confirmation,
        selected_xgboost,
        development_config.research.random_seed,
    )
    primary_predictions["xgboost_expanded"] = xgb_result
    actual, predicted, training_mean, train_rows = xgb_result
    model_rows.append(
        _metric_row(
            "BTCUSDT", 5_000, "xgboost", "expanded", train_rows, actual, predicted, training_mean
        )
    )
    daily_rows.extend(
        _daily_metrics(confirmation, actual, predicted, training_mean, "xgboost", "expanded")
    )

    for horizon_ms in HORIZONS_MS:
        target = f"target_spot_log_return_{horizon_ms}ms"
        for scope, features in (("baseline", BASELINE_FEATURES), ("expanded", EXPANDED_FEATURES)):
            result = _fit_ridge(
                development,
                confirmation,
                features,
                target,
                development_config.research.ridge_alpha,
            )
            horizon_rows.append(
                _metric_row("BTCUSDT", horizon_ms, "ridge", scope, result[3], *result[:3])
            )

    placebo_rows = [
        _placebo_result(development, confirmation, development_config.research.ridge_alpha)
    ]
    ridge_actual, ridge_prediction, ridge_mean, _ = primary_predictions["ridge_expanded"]
    regime_rows = _regime_rows(
        development, confirmation, ridge_actual, ridge_prediction, ridge_mean
    )

    bootstrap_rows = [
        _bootstrap_row(
            "ridge_baseline_minus_expanded_mse",
            [row for row in daily_rows if row["model"] == "ridge" and row["scope"] == "baseline"],
            [row for row in daily_rows if row["model"] == "ridge" and row["scope"] == "expanded"],
            confirmation_config.research.bootstrap_replicates,
            confirmation_config.research.random_seed,
        ),
        _bootstrap_row(
            "ridge_minus_xgboost_mse",
            [row for row in daily_rows if row["model"] == "ridge" and row["scope"] == "expanded"],
            [row for row in daily_rows if row["model"] == "xgboost"],
            confirmation_config.research.bootstrap_replicates,
            confirmation_config.research.random_seed,
        ),
    ]

    del development, confirmation
    gc.collect()
    eth_development = _load_features(development_config, "ETHUSDT")
    eth_confirmation = _load_features(confirmation_config, "ETHUSDT")
    for scope, features in (("baseline", BASELINE_FEATURES), ("expanded", EXPANDED_FEATURES)):
        result = _fit_ridge(
            eth_development,
            eth_confirmation,
            features,
            PRIMARY_TARGET,
            development_config.research.ridge_alpha,
        )
        eth_rows.append(_metric_row("ETHUSDT", 5_000, "ridge", scope, result[3], *result[:3]))
    del eth_development, eth_confirmation
    gc.collect()

    ridge_metric = next(
        row for row in model_rows if row["model"] == "ridge" and row["scope"] == "expanded"
    )
    xgb_metric = next(row for row in model_rows if row["model"] == "xgboost")
    xgb_bootstrap = next(
        row for row in bootstrap_rows if row["comparison"] == "ridge_minus_xgboost_mse"
    )
    xgb_selected = bool(
        xgb_metric["oos_r2"] >= ridge_metric["oos_r2"] + 0.001 and xgb_bootstrap["lower"] > 0
    )
    selected_model = "xgboost" if xgb_selected else "ridge"
    selected_parameters: dict[str, Any] = (
        {**XGBOOST_FIXED, **selected_xgboost}
        if xgb_selected
        else {"alpha": development_config.research.ridge_alpha}
    )

    outputs = {
        "model_metrics": reports / "phase5_model_metrics.csv",
        "daily_metrics": reports / "phase5_daily_metrics.csv",
        "xgboost_tuning": reports / "phase5_xgboost_tuning.csv",
        "bootstrap": reports / "phase5_bootstrap.csv",
        "regimes": reports / "phase5_regimes.csv",
        "horizons": reports / "phase5_horizons.csv",
        "eth_replication": reports / "phase5_eth_replication.csv",
        "placebo": reports / "phase5_placebo.csv",
        "failures": reports / "phase5_failures.csv",
    }
    for name, rows in (
        ("model_metrics", model_rows),
        ("daily_metrics", daily_rows),
        ("xgboost_tuning", tuning_rows),
        ("bootstrap", bootstrap_rows),
        ("regimes", regime_rows),
        ("horizons", horizon_rows),
        ("eth_replication", eth_rows),
        ("placebo", placebo_rows),
    ):
        _write_csv(outputs[name], rows)
    _write_csv(
        outputs["failures"],
        failures,
        {"check": pl.String, "error_type": pl.String, "message": pl.String},
    )

    ridge_baseline = next(
        row for row in model_rows if row["model"] == "ridge" and row["scope"] == "baseline"
    )
    eth_baseline = next(row for row in eth_rows if row["scope"] == "baseline")
    eth_expanded = next(row for row in eth_rows if row["scope"] == "expanded")
    placebo = placebo_rows[0]
    summary_path = reports / "phase5_summary.md"
    summary_path.write_text(
        "# Phase 5 confirmation and robustness summary\n\n"
        "These are prespecified confirmation results, not final-holdout or trading results.\n\n"
        f"- Selected final model class by the frozen rule: `{selected_model}` / `expanded`\n"
        f"- BTC Ridge OOS R-squared, baseline / expanded: "
        f"{ridge_baseline['oos_r2']:.6g} / {ridge_metric['oos_r2']:.6g}\n"
        f"- BTC XGBoost expanded OOS R-squared: {xgb_metric['oos_r2']:.6g}\n"
        f"- Ridge-minus-XGBoost daily MSE bootstrap 95% interval: "
        f"[{xgb_bootstrap['lower']:.3e}, {xgb_bootstrap['upper']:.3e}]\n"
        f"- Selected XGBoost candidate: {json.dumps(selected_xgboost, sort_keys=True)}\n"
        f"- BTC placebo-expanded OOS R-squared: {placebo['oos_r2']:.6g}\n"
        f"- ETH Ridge OOS R-squared, baseline / expanded: "
        f"{eth_baseline['oos_r2']:.6g} / {eth_expanded['oos_r2']:.6g}\n"
        f"- Recorded failures: {len(failures)}\n\n"
        "The final holdout remains sealed. Statistical predictability is not evidence of "
        "economic tradability; costs, latency and execution remain deferred to Phase 6.\n",
        encoding="utf-8",
    )
    outputs["summary"] = summary_path

    manifest_dir = confirmation_config.data.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    final_spec_payload = {
        "status": "frozen_after_phase5_before_final_holdout",
        "model": selected_model,
        "scope": "expanded",
        "target": PRIMARY_TARGET,
        "features": EXPANDED_FEATURES,
        "parameters": selected_parameters,
        "preprocessing": (
            "development-plus-confirmation fit; median imputation and standardisation "
            "for Ridge only"
        ),
        "training_dates": {"start": "2025-01-02", "end": "2025-02-20"},
        "xgboost_training_subsample_seconds": 5 if xgb_selected else None,
        "final_evaluation_dates": {"start": "2025-02-21", "end": "2025-03-02"},
        "final_holdout_status": "sealed",
    }
    final_spec_hash = hashlib.sha256(canonical_json(final_spec_payload)).hexdigest()
    final_spec = {**final_spec_payload, "specification_hash": final_spec_hash}
    final_spec_path = manifest_dir / "final-model-specification.json"
    final_spec_path.write_text(
        json.dumps(final_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    feature_manifests = {
        name: json.loads((manifest_dir / f"{name}-features.json").read_text())["manifest_hash"]
        for name in (development_config.name, confirmation_config.name)
    }
    payload = {
        "phase": 5,
        "development_config": development_config.name,
        "confirmation_config": confirmation_config.name,
        "feature_manifest_hashes": feature_manifests,
        "protocol_sha256": sha256_file(Path("docs/phase5_protocol.md")),
        "xgboost_grid": XGBOOST_GRID,
        "selected_xgboost_parameters": selected_xgboost,
        "selected_final_model": selected_model,
        "final_specification_hash": final_spec_hash,
        "outputs": {name: sha256_file(path) for name, path in outputs.items()},
        "failures": len(failures),
        "random_seed": confirmation_config.research.random_seed,
    }
    manifest_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    manifest = {**payload, "manifest_hash": manifest_hash}
    manifest_path = manifest_dir / f"{confirmation_config.name}-phase5.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "selected_model": selected_model,
        "selected_xgboost": selected_xgboost,
        "manifest_hash": manifest_hash,
        "failures": len(failures),
    }
