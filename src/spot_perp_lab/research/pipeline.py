"""Frozen Phase 4 walk-forward research pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from spot_perp_lab.config import AppConfig
from spot_perp_lab.data.checksums import sha256_file
from spot_perp_lab.data.manifest import canonical_json
from spot_perp_lab.data.pipeline import SealedHoldoutError
from spot_perp_lab.features.pipeline import feature_output_path
from spot_perp_lab.research.baselines import (
    BASELINE_FEATURES,
    EXPANDED_FEATURES,
    HAC_BASELINE_FEATURES,
    HAC_EXPANDED_FEATURES,
    training_mean_predictions,
)
from spot_perp_lab.research.evaluation import classification_metrics, regression_metrics
from spot_perp_lab.research.inference import hac_regression, paired_day_bootstrap
from spot_perp_lab.research.models import classification_pipeline, regression_pipeline
from spot_perp_lab.research.splits import expanding_day_folds, split_frame

REGRESSION_TARGET = "target_spot_log_return_5000ms"
CLASSIFICATION_TARGET = "target_spot_direction_5000ms"


def _load_development_features(config: AppConfig) -> pl.DataFrame:
    days = pl.date_range(
        config.data.dates.start, config.data.dates.end, interval="1d", eager=True
    ).to_list()
    paths = [
        feature_output_path(
            config.features.output_dir, config.name, config.research.primary_symbol, day
        )
        for day in days
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} feature partitions; first: {missing[0]}")
    columns = [
        "date",
        "decision_time_ns",
        *EXPANDED_FEATURES,
        REGRESSION_TARGET,
        CLASSIFICATION_TARGET,
    ]
    return pl.concat(
        [pl.read_parquet(path, columns=columns, hive_partitioning=False) for path in paths]
    ).sort("decision_time_ns")


def _fold_row(
    *,
    fold: int,
    task: str,
    model: str,
    scope: str,
    train_rows: int,
    evaluation_rows: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold,
        "task": task,
        "model": model,
        "scope": scope,
        "train_rows": train_rows,
        "evaluation_rows": evaluation_rows,
        "oos_r2": None,
        "mse": None,
        "mae": None,
        "pearson_ic": None,
        "rank_ic": None,
        "roc_auc": None,
        "pr_auc": None,
        "brier": None,
        "accuracy": None,
        "class_balance": None,
    }
    row.update(metrics)
    return row


def _daily_rows(
    *,
    dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    training_mean: float,
    fold: int,
    task: str,
    model: str,
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in sorted(set(dates.tolist())):
        mask = dates == day
        metrics = (
            regression_metrics(actual[mask], predicted[mask], training_mean)
            if task == "regression"
            else classification_metrics(actual[mask], predicted[mask])
        )
        rows.append(
            {
                "fold": fold,
                "date": day,
                "task": task,
                "model": model,
                "scope": scope,
                **metrics,
            }
        )
    return rows


def _decile_rows(
    actual: np.ndarray,
    predicted: np.ndarray,
    fold: int,
    task: str,
    model: str,
    scope: str,
) -> list[dict[str, Any]]:
    order = np.argsort(predicted, kind="stable")
    deciles = np.empty(predicted.size, dtype=np.int8)
    deciles[order] = np.minimum(np.arange(predicted.size) * 10 // predicted.size + 1, 10)
    return [
        {
            "fold": fold,
            "task": task,
            "model": model,
            "scope": scope,
            "decile": decile,
            "observations": int(np.sum(mask := deciles == decile)),
            "mean_prediction": float(np.mean(predicted[mask])),
            "mean_actual": float(np.mean(actual[mask])),
        }
        for decile in range(1, 11)
    ]


def _write_csv(
    path: Path, rows: list[dict[str, Any]], schema: dict[str, Any] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema or {})
    frame.write_csv(path)


def run_phase4(config: AppConfig) -> dict[str, Any]:
    """Run the frozen Phase 4 development procedure and write versioned reports."""

    if config.research.holdout_status == "sealed":
        raise SealedHoldoutError(f"configuration {config.name!r} is sealed")
    if config.research.design_status != "frozen":
        raise ValueError("Phase 4 requires a frozen research design")
    frame = _load_development_features(config)
    dates = tuple(sorted(frame["date"].unique().to_list()))
    folds = expanding_day_folds(
        dates, config.research.initial_train_days, config.research.test_days
    )
    fold_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for fold in folds:
        train, evaluation = split_frame(frame, fold, config.research.purge_seconds)
        train_regression = train.filter(pl.col(REGRESSION_TARGET).is_not_null())
        eval_regression = evaluation.filter(pl.col(REGRESSION_TARGET).is_not_null())
        y_train = train_regression[REGRESSION_TARGET].to_numpy().astype(np.float64)
        y_eval = eval_regression[REGRESSION_TARGET].to_numpy().astype(np.float64)
        eval_dates = eval_regression["date"].to_numpy()
        mean_prediction, training_mean = training_mean_predictions(y_train, y_eval.size)
        references = {
            "training_mean": mean_prediction,
            "zero_return": np.zeros(y_eval.size, dtype=np.float64),
        }
        for model, predictions in references.items():
            metrics = regression_metrics(y_eval, predictions, training_mean)
            fold_rows.append(
                _fold_row(
                    fold=fold.fold,
                    task="regression",
                    model=model,
                    scope="reference",
                    train_rows=y_train.size,
                    evaluation_rows=y_eval.size,
                    metrics=metrics,
                )
            )
            daily_rows.extend(
                _daily_rows(
                    dates=eval_dates,
                    actual=y_eval,
                    predicted=predictions,
                    training_mean=training_mean,
                    fold=fold.fold,
                    task="regression",
                    model=model,
                    scope="reference",
                )
            )

        for model in ("linear", "ridge"):
            for scope, features in (
                ("baseline", BASELINE_FEATURES),
                ("expanded", EXPANDED_FEATURES),
            ):
                try:
                    estimator = regression_pipeline(model, config.research.ridge_alpha)
                    x_train = train_regression.select(features).to_numpy()
                    x_eval = eval_regression.select(features).to_numpy()
                    estimator.fit(x_train, y_train)
                    prediction = estimator.predict(x_eval).astype(np.float64)
                    metrics = regression_metrics(y_eval, prediction, training_mean)
                    fold_rows.append(
                        _fold_row(
                            fold=fold.fold,
                            task="regression",
                            model=model,
                            scope=scope,
                            train_rows=y_train.size,
                            evaluation_rows=y_eval.size,
                            metrics=metrics,
                        )
                    )
                    daily_rows.extend(
                        _daily_rows(
                            dates=eval_dates,
                            actual=y_eval,
                            predicted=prediction,
                            training_mean=training_mean,
                            fold=fold.fold,
                            task="regression",
                            model=model,
                            scope=scope,
                        )
                    )
                    decile_rows.extend(
                        _decile_rows(y_eval, prediction, fold.fold, "regression", model, scope)
                    )
                except Exception as error:
                    failures.append(
                        {
                            "fold": fold.fold,
                            "task": "regression",
                            "model": model,
                            "scope": scope,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )

        train_classification = train.filter(pl.col(CLASSIFICATION_TARGET).is_not_null())
        eval_classification = evaluation.filter(pl.col(CLASSIFICATION_TARGET).is_not_null())
        y_train_class = train_classification[CLASSIFICATION_TARGET].to_numpy().astype(np.int8)
        y_eval_class = eval_classification[CLASSIFICATION_TARGET].to_numpy().astype(np.int8)
        eval_class_dates = eval_classification["date"].to_numpy()
        for scope, features in (
            ("baseline", BASELINE_FEATURES),
            ("expanded", EXPANDED_FEATURES),
        ):
            try:
                estimator = classification_pipeline(
                    config.research.logistic_c, config.research.random_seed
                )
                x_train = train_classification.select(features).to_numpy()
                x_eval = eval_classification.select(features).to_numpy()
                estimator.fit(x_train, y_train_class)
                probability = estimator.predict_proba(x_eval)[:, 1].astype(np.float64)
                metrics = classification_metrics(y_eval_class, probability)
                fold_rows.append(
                    _fold_row(
                        fold=fold.fold,
                        task="classification",
                        model="logistic",
                        scope=scope,
                        train_rows=y_train_class.size,
                        evaluation_rows=y_eval_class.size,
                        metrics=metrics,
                    )
                )
                daily_rows.extend(
                    _daily_rows(
                        dates=eval_class_dates,
                        actual=y_eval_class,
                        predicted=probability,
                        training_mean=float(np.mean(y_train_class)),
                        fold=fold.fold,
                        task="classification",
                        model="logistic",
                        scope=scope,
                    )
                )
                decile_rows.extend(
                    _decile_rows(
                        y_eval_class, probability, fold.fold, "classification", "logistic", scope
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "fold": fold.fold,
                        "task": "classification",
                        "model": "logistic",
                        "scope": scope,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
        del train, evaluation
        gc.collect()

    bootstrap_rows: list[dict[str, Any]] = []
    for model in ("linear", "ridge"):
        baseline = {
            (row["fold"], row["date"]): row["mse"]
            for row in daily_rows
            if row["task"] == "regression" and row["model"] == model and row["scope"] == "baseline"
        }
        expanded = {
            (row["fold"], row["date"]): row["mse"]
            for row in daily_rows
            if row["task"] == "regression" and row["model"] == model and row["scope"] == "expanded"
        }
        keys = sorted(baseline.keys() & expanded.keys())
        improvements = np.array([baseline[key] - expanded[key] for key in keys], dtype=np.float64)
        if improvements.size >= 2:
            interval = paired_day_bootstrap(
                improvements,
                config.research.bootstrap_replicates,
                config.research.random_seed,
            )
            bootstrap_rows.append(
                {"model": model, "metric": "daily_mse_improvement", **asdict(interval)}
            )

    hac_frame = frame.filter(
        pl.col(REGRESSION_TARGET).is_not_null() & (pl.col("decision_time_ns") % 5_000_000_000 == 0)
    )
    hac_rows: list[dict[str, Any]] = []
    for scope, hac_features in (
        ("baseline", HAC_BASELINE_FEATURES),
        ("expanded", HAC_EXPANDED_FEATURES),
    ):
        hac_rows.extend(
            hac_regression(
                hac_frame.select(hac_features).to_numpy(),
                hac_frame[REGRESSION_TARGET].to_numpy().astype(np.float64),
                hac_features,
                config.research.hac_max_lags,
                scope,
            )
        )

    reports = config.research.reports_dir
    outputs = {
        "fold_metrics": reports / "phase4_fold_metrics.csv",
        "daily_metrics": reports / "phase4_daily_metrics.csv",
        "deciles": reports / "phase4_deciles.csv",
        "bootstrap": reports / "phase4_bootstrap.csv",
        "hac": reports / "phase4_hac.csv",
        "failures": reports / "phase4_failures.csv",
    }
    _write_csv(outputs["fold_metrics"], fold_rows)
    _write_csv(outputs["daily_metrics"], daily_rows)
    _write_csv(outputs["deciles"], decile_rows)
    _write_csv(outputs["bootstrap"], bootstrap_rows)
    _write_csv(outputs["hac"], hac_rows)
    _write_csv(
        outputs["failures"],
        failures,
        {
            "fold": pl.Int64,
            "task": pl.String,
            "model": pl.String,
            "scope": pl.String,
            "error_type": pl.String,
            "message": pl.String,
        },
    )

    candidate = (
        pl.DataFrame(fold_rows)
        .filter((pl.col("task") == "regression") & pl.col("model").is_in(["linear", "ridge"]))
        .group_by(["model", "scope"])
        .agg(pl.col("oos_r2").mean().alias("mean_fold_oos_r2"))
        .sort("mean_fold_oos_r2", descending=True)
    )
    preferred = candidate.row(0, named=True)
    preferred_baseline = candidate.filter(
        (pl.col("model") == preferred["model"]) & (pl.col("scope") == "baseline")
    ).row(0, named=True)
    preferred_bootstrap = next(
        (row for row in bootstrap_rows if row["model"] == preferred["model"]), None
    )
    preferred_daily_baseline = {
        (row["fold"], row["date"]): row["mse"]
        for row in daily_rows
        if row["task"] == "regression"
        and row["model"] == preferred["model"]
        and row["scope"] == "baseline"
    }
    preferred_daily_expanded = {
        (row["fold"], row["date"]): row["mse"]
        for row in daily_rows
        if row["task"] == "regression"
        and row["model"] == preferred["model"]
        and row["scope"] == "expanded"
    }
    paired_days = sorted(preferred_daily_baseline.keys() & preferred_daily_expanded.keys())
    positive_days = sum(
        preferred_daily_baseline[key] > preferred_daily_expanded[key] for key in paired_days
    )
    classification_summary = (
        pl.DataFrame(fold_rows)
        .filter((pl.col("task") == "classification") & (pl.col("model") == "logistic"))
        .group_by("scope")
        .agg(
            pl.col("roc_auc").mean().alias("mean_auc"),
            pl.col("pr_auc").mean().alias("mean_pr_auc"),
            pl.col("brier").mean().alias("mean_brier"),
        )
    )
    classification = {row["scope"]: row for row in classification_summary.iter_rows(named=True)}
    summary_path = reports / "phase4_summary.md"
    interval_text = (
        f"[{preferred_bootstrap['lower']:.3e}, {preferred_bootstrap['upper']:.3e}]"
        if preferred_bootstrap
        else "unavailable because paired model results failed"
    )
    summary_path.write_text(
        "# Phase 4 development summary\n\n"
        "These are walk-forward development results, not final-holdout or trading results.\n\n"
        f"- Development rows: {frame.height:,}\n"
        f"- Walk-forward folds: {len(folds)}\n"
        f"- Preferred fixed linear specification by the frozen rule: "
        f"`{preferred['model']}` / `{preferred['scope']}`\n"
        f"- Mean fold OOS R-squared: {preferred['mean_fold_oos_r2']:.6g}\n"
        f"- Same-model baseline mean fold OOS R-squared: "
        f"{preferred_baseline['mean_fold_oos_r2']:.6g}\n"
        f"- Expanded-minus-baseline mean OOS R-squared: "
        f"{preferred['mean_fold_oos_r2'] - preferred_baseline['mean_fold_oos_r2']:.6g}\n"
        f"- Paired daily MSE-improvement 95% bootstrap interval: {interval_text}\n"
        f"- Evaluation days with lower expanded-model MSE: {positive_days}/{len(paired_days)}\n"
        f"- Mean logistic ROC AUC, baseline / expanded: "
        f"{classification['baseline']['mean_auc']:.6g} / "
        f"{classification['expanded']['mean_auc']:.6g}\n"
        f"- Mean logistic Brier score, baseline / expanded: "
        f"{classification['baseline']['mean_brier']:.6g} / "
        f"{classification['expanded']['mean_brier']:.6g}\n"
        f"- Recorded model failures: {len(failures)}\n\n"
        "Statistical predictability, if any, is not evidence of economic tradability. "
        "Costs, latency and execution are deferred to Phase 6.\n",
        encoding="utf-8",
    )
    outputs["summary"] = summary_path

    feature_manifest_path = config.data.manifest_dir / f"{config.name}-features.json"
    feature_manifest_hash = (
        json.loads(feature_manifest_path.read_text())["manifest_hash"]
        if feature_manifest_path.exists()
        else None
    )
    payload = {
        "phase": 4,
        "config_name": config.name,
        "feature_manifest_hash": feature_manifest_hash,
        "folds": [asdict(fold) for fold in folds],
        "baseline_features": BASELINE_FEATURES,
        "expanded_features": EXPANDED_FEATURES,
        "outputs": {name: sha256_file(path) for name, path in outputs.items()},
        "failures": len(failures),
        "random_seed": config.research.random_seed,
    }
    manifest_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    manifest = {**payload, "manifest_hash": manifest_hash}
    config.data.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.data.manifest_dir / f"{config.name}-phase4.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "folds": len(folds),
        "rows": frame.height,
        "failures": len(failures),
        "preferred_model": preferred["model"],
        "preferred_scope": preferred["scope"],
        "manifest_hash": manifest_hash,
    }
