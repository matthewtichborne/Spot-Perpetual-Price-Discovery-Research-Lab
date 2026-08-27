"""Frozen Phase 6 signal, execution, cost, and portfolio pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from spot_perp_lab.config import AppConfig, load_config
from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.checksums import sha256_file
from spot_perp_lab.data.manifest import canonical_json
from spot_perp_lab.data.normalise import processed_path
from spot_perp_lab.data.pipeline import SealedHoldoutError
from spot_perp_lab.execution.config import Phase6Config
from spot_perp_lab.execution.engine import apply_roundtrip_cost, execute_non_overlapping
from spot_perp_lab.execution.metrics import daily_pnl, performance_metrics, portfolio_ledger
from spot_perp_lab.research.baselines import EXPANDED_FEATURES
from spot_perp_lab.research.models import regression_pipeline, xgboost_regressor
from spot_perp_lab.research.phase5 import PRIMARY_TARGET, _load_features


def _validation_frames(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    boundary_ns = int(datetime(2025, 1, 27, tzinfo=UTC).timestamp()) * 1_000_000_000
    eligible = frame.filter(pl.col(PRIMARY_TARGET).is_not_null())
    train = eligible.filter(pl.col("decision_time_ns") < boundary_ns - 10_000_000_000)
    validation = eligible.filter(pl.col("decision_time_ns") >= boundary_ns + 10_000_000_000)
    if not train.height or not validation.height:
        raise ValueError("Phase 6 validation calibration split is empty")
    return train, validation


def _signal_frame(
    development_config: AppConfig,
    evaluation_config: AppConfig,
    symbol: str,
    final_specification: dict[str, Any],
    config: Phase6Config,
) -> tuple[pl.DataFrame, float, float]:
    development = _load_features(development_config, symbol)
    evaluation = _load_features(evaluation_config, symbol).filter(
        pl.col(PRIMARY_TARGET).is_not_null()
    )
    tuning_train, validation = _validation_frames(development)
    if symbol == "BTCUSDT":
        parameters: dict[str, int | float] = {
            "max_depth": int(final_specification["parameters"]["max_depth"]),
            "n_estimators": int(final_specification["parameters"]["n_estimators"]),
        }
        tuning_train = tuning_train.filter(pl.col("decision_time_ns") % 5_000_000_000 == 0)
        validation_model = xgboost_regressor(parameters, config.random_seed)
        validation_model.fit(
            tuning_train.select(EXPANDED_FEATURES).to_numpy(),
            tuning_train[PRIMARY_TARGET].to_numpy(),
        )
        validation_prediction = validation_model.predict(
            validation.select(EXPANDED_FEATURES).to_numpy()
        ).astype(np.float64)
        full_train = development.filter(
            pl.col(PRIMARY_TARGET).is_not_null() & (pl.col("decision_time_ns") % 5_000_000_000 == 0)
        )
        model = xgboost_regressor(parameters, config.random_seed)
    else:
        validation_model = regression_pipeline("ridge", development_config.research.ridge_alpha)
        validation_model.fit(
            tuning_train.select(EXPANDED_FEATURES).to_numpy(),
            tuning_train[PRIMARY_TARGET].to_numpy(),
        )
        validation_prediction = validation_model.predict(
            validation.select(EXPANDED_FEATURES).to_numpy()
        ).astype(np.float64)
        full_train = development.filter(pl.col(PRIMARY_TARGET).is_not_null())
        model = regression_pipeline("ridge", development_config.research.ridge_alpha)
    threshold = config.signal_threshold_sigma * float(np.std(validation_prediction))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"non-positive signal threshold for {symbol}")
    model.fit(
        full_train.select(EXPANDED_FEATURES).to_numpy(),
        full_train[PRIMARY_TARGET].to_numpy(),
    )
    prediction = model.predict(evaluation.select(EXPANDED_FEATURES).to_numpy()).astype(np.float64)
    development_volatility = float(
        np.median(development["spot_realised_volatility_5000ms"].to_numpy())
    )
    signals = evaluation.select(
        "date", "decision_time_ns", "spot_realised_volatility_5000ms"
    ).with_columns(pl.Series("prediction", prediction))
    return signals, threshold, development_volatility


def _sized_signals(
    signals: pl.DataFrame,
    sizing: str,
    development_volatility: float,
    config: Phase6Config,
) -> pl.DataFrame:
    if sizing == "fixed":
        size = pl.lit(config.maximum_gross_exposure)
    elif sizing == "inverse_volatility":
        size = (
            (
                pl.lit(development_volatility)
                / pl.col("spot_realised_volatility_5000ms").clip(lower_bound=1e-12)
            )
            .clip(
                lower_bound=config.inverse_volatility_minimum_multiplier,
                upper_bound=config.inverse_volatility_maximum_multiplier,
            )
            .clip(upper_bound=config.maximum_gross_exposure)
        )
    else:
        raise ValueError(f"unknown sizing rule: {sizing}")
    return signals.with_columns(size.alias("size"))


def _execute_symbol(
    evaluation_config: AppConfig,
    symbol: str,
    signals: pl.DataFrame,
    threshold: float,
    development_volatility: float,
    config: Phase6Config,
) -> tuple[dict[tuple[int, str], pl.DataFrame], list[dict[str, Any]]]:
    trade_sets: dict[tuple[int, str], pl.DataFrame] = {}
    diagnostics: list[dict[str, Any]] = []
    sizing_rules = ("fixed", "inverse_volatility")
    for sizing in sizing_rules:
        sized = _sized_signals(signals, sizing, development_volatility, config)
        latencies = config.latencies_ms if sizing == "fixed" else (config.primary_latency_ms,)
        for latency_ms in latencies:
            day_frames: list[pl.DataFrame] = []
            skipped_entries = 0
            skipped_exits = 0
            for day in sorted(sized["date"].unique().to_list()):
                spot = pl.read_parquet(
                    processed_path(
                        evaluation_config.data.processed_dir,
                        MarketType.SPOT,
                        symbol,
                        datetime.strptime(day, "%Y-%m-%d").date(),
                    ),
                    columns=["event_time_ns", "price"],
                    hive_partitioning=False,
                )
                result = execute_non_overlapping(
                    sized.filter(pl.col("date") == day),
                    spot,
                    symbol=symbol,
                    day=day,
                    latency_ms=latency_ms,
                    holding_seconds=config.holding_seconds,
                    threshold=threshold,
                )
                day_frames.append(result.trades)
                skipped_entries += result.skipped_entries
                skipped_exits += result.skipped_exits
            trade_sets[(latency_ms, sizing)] = pl.concat(day_frames)
            diagnostics.append(
                {
                    "symbol": symbol,
                    "latency_ms": latency_ms,
                    "sizing": sizing,
                    "trades": trade_sets[(latency_ms, sizing)].height,
                    "skipped_entries": skipped_entries,
                    "skipped_exits": skipped_exits,
                }
            )
    return trade_sets, diagnostics


def _risk_weights(volatility: dict[str, float]) -> dict[str, float]:
    inverse = {symbol: 1.0 / value for symbol, value in volatility.items()}
    total = sum(inverse.values())
    return {symbol: value / total for symbol, value in inverse.items()}


def _metric_row(
    entity: str,
    latency_ms: int,
    cost_bps: float,
    sizing: str,
    trades: pl.DataFrame,
    dates: tuple[str, ...],
    config: Phase6Config,
) -> dict[str, Any]:
    daily = daily_pnl(trades, dates)
    return {
        "entity": entity,
        "latency_ms": latency_ms,
        "roundtrip_cost_bps": cost_bps,
        "sizing": sizing,
        **performance_metrics(trades, daily, config.annualisation_days),
    }


def _write_cost_latency_figure(rows: list[dict[str, Any]], path: Path) -> None:
    combined = [row for row in rows if row["entity"] == "combined" and row["sizing"] == "fixed"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for cost in sorted({float(row["roundtrip_cost_bps"]) for row in combined}):
        selected = sorted(
            (row for row in combined if float(row["roundtrip_cost_bps"]) == cost),
            key=lambda row: int(row["latency_ms"]),
        )
        axis.plot(
            [int(row["latency_ms"]) for row in selected],
            [float(row["net_pnl"]) for row in selected],
            marker="o",
            label=f"{cost:g} bps",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Latency (ms)", ylabel="Combined net P&L", title="Phase 6 cost/latency sensitivity"
    )
    axis.legend(title="Round trip", ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_phase6(config: Phase6Config) -> dict[str, Any]:
    """Run the frozen economic analysis on the open confirmation sample."""

    development_config = load_config(config.development_config)
    evaluation_config = load_config(config.evaluation_config)
    if development_config.research.holdout_status == "sealed":
        raise SealedHoldoutError(f"configuration {development_config.name!r} is sealed")
    if evaluation_config.research.holdout_status == "sealed":
        raise SealedHoldoutError(f"configuration {evaluation_config.name!r} is sealed")
    if evaluation_config.data.dates.end.isoformat() != "2025-02-20":
        raise ValueError("Phase 6 may only use the frozen confirmation period")
    final_specification = json.loads(
        (evaluation_config.data.manifest_dir / "final-model-specification.json").read_text()
    )
    if final_specification["model"] != "xgboost" or final_specification["scope"] != "expanded":
        raise ValueError("Phase 6 final model specification does not match Phase 5")

    symbols = ("BTCUSDT", "ETHUSDT")
    thresholds: dict[str, float] = {}
    volatility: dict[str, float] = {}
    all_trades: dict[str, dict[tuple[int, str], pl.DataFrame]] = {}
    diagnostics: list[dict[str, Any]] = []
    for symbol in symbols:
        signals, thresholds[symbol], volatility[symbol] = _signal_frame(
            development_config, evaluation_config, symbol, final_specification, config
        )
        all_trades[symbol], symbol_diagnostics = _execute_symbol(
            evaluation_config,
            symbol,
            signals,
            thresholds[symbol],
            volatility[symbol],
            config,
        )
        diagnostics.extend(symbol_diagnostics)

    dates = tuple(
        (evaluation_config.data.dates.start + timedelta(days=offset)).isoformat()
        for offset in range(
            (evaluation_config.data.dates.end - evaluation_config.data.dates.start).days + 1
        )
    )
    weights = _risk_weights(volatility)
    sensitivity_rows: list[dict[str, Any]] = []
    portfolio_sets: dict[tuple[int, float], pl.DataFrame] = {}
    for latency_ms in config.latencies_ms:
        for cost_bps in config.cost_sensitivity_roundtrip_bps:
            for symbol in symbols:
                costed = apply_roundtrip_cost(all_trades[symbol][(latency_ms, "fixed")], cost_bps)
                sensitivity_rows.append(
                    _metric_row(symbol, latency_ms, cost_bps, "fixed", costed, dates, config)
                )
            portfolio = portfolio_ledger(
                {symbol: all_trades[symbol][(latency_ms, "fixed")] for symbol in symbols},
                weights,
                cost_bps,
                config.daily_loss_limit,
            )
            portfolio_sets[(latency_ms, cost_bps)] = portfolio
            sensitivity_rows.append(
                _metric_row("combined", latency_ms, cost_bps, "fixed", portfolio, dates, config)
            )
    for cost_bps in config.cost_sensitivity_roundtrip_bps:
        for symbol in symbols:
            costed = apply_roundtrip_cost(
                all_trades[symbol][(config.primary_latency_ms, "inverse_volatility")],
                cost_bps,
            )
            sensitivity_rows.append(
                _metric_row(
                    symbol,
                    config.primary_latency_ms,
                    cost_bps,
                    "inverse_volatility",
                    costed,
                    dates,
                    config,
                )
            )

    primary_cost = config.reference_roundtrip_cost_bps
    primary_asset_ledgers = {
        symbol: apply_roundtrip_cost(
            all_trades[symbol][(config.primary_latency_ms, "fixed")], primary_cost
        )
        for symbol in symbols
    }
    primary_portfolio = portfolio_sets[(config.primary_latency_ms, primary_cost)]
    daily_frames: list[pl.DataFrame] = []
    for entity, ledger in (*primary_asset_ledgers.items(), ("combined", primary_portfolio)):
        daily = daily_pnl(ledger, dates).with_columns(pl.lit(entity).alias("entity"))
        if not np.isclose(
            float(daily["gross_pnl"].sum()),
            float(ledger["gross_pnl"].sum()),
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError(f"gross daily P&L does not reconcile for {entity}")
        if not np.isclose(
            float(daily["net_pnl"].sum()),
            float(ledger["net_pnl"].sum()),
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError(f"net daily P&L does not reconcile for {entity}")
        daily_frames.append(daily)
    primary_daily = pl.concat(daily_frames).select(
        "entity", "date", "trades", "gross_pnl", "cost_pnl", "net_pnl"
    )

    regime_rows: list[dict[str, Any]] = []
    for symbol, ledger in primary_asset_ledgers.items():
        for regime, expression in (
            ("low", pl.col("volatility") < volatility[symbol]),
            ("high", pl.col("volatility") >= volatility[symbol]),
        ):
            subset = ledger.filter(expression)
            regime_rows.append(
                {
                    "symbol": symbol,
                    "regime": regime,
                    "development_volatility_threshold": volatility[symbol],
                    **performance_metrics(
                        subset, daily_pnl(subset, dates), config.annualisation_days
                    ),
                }
            )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.ledger_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sensitivity": config.reports_dir / "phase6_sensitivity.csv",
        "daily": config.reports_dir / "phase6_daily.csv",
        "regimes": config.reports_dir / "phase6_regimes.csv",
        "diagnostics": config.reports_dir / "phase6_execution_diagnostics.csv",
        "summary": config.reports_dir / "phase6_summary.md",
        "figure": config.reports_dir.parent / "figures" / "phase6-cost-latency.png",
    }
    pl.DataFrame(sensitivity_rows).write_csv(outputs["sensitivity"])
    primary_daily.write_csv(outputs["daily"])
    pl.DataFrame(regime_rows).write_csv(outputs["regimes"])
    pl.DataFrame(diagnostics).write_csv(outputs["diagnostics"])
    asset_ledger_path = config.ledger_dir / "phase6-primary-asset-trades.parquet"
    portfolio_ledger_path = config.ledger_dir / "phase6-primary-portfolio-trades.parquet"
    pl.concat(list(primary_asset_ledgers.values())).write_parquet(asset_ledger_path)
    primary_portfolio.write_parquet(portfolio_ledger_path)
    _write_cost_latency_figure(sensitivity_rows, outputs["figure"])

    primary_rows = {
        row["entity"]: row
        for row in sensitivity_rows
        if row["latency_ms"] == config.primary_latency_ms
        and row["roundtrip_cost_bps"] == primary_cost
        and row["sizing"] == "fixed"
    }
    combined = primary_rows["combined"]
    conclusion = (
        "survives the registered cost assumption"
        if float(combined["net_pnl"]) > 0
        else "does not survive the registered cost assumption"
    )
    outputs["summary"].write_text(
        "# Phase 6 execution and portfolio summary\n\n"
        "These are confirmation-sample cost/latency sensitivities, not exact fill or "
        "final-holdout results.\n\n"
        f"- Primary latency: {config.primary_latency_ms} ms\n"
        f"- Primary all-in round-trip cost: {primary_cost:g} bps\n"
        f"- BTC / ETH portfolio risk weights: {weights['BTCUSDT']:.4f} / "
        f"{weights['ETHUSDT']:.4f}\n"
        f"- Combined trades: {combined['trades']:,}\n"
        f"- Combined gross P&L: {combined['gross_pnl']:.6g}\n"
        f"- Combined net P&L: {combined['net_pnl']:.6g}\n"
        f"- Combined annualised daily net Sharpe: "
        f"{combined['annualised_daily_sharpe']:.6g}\n"
        f"- Combined maximum drawdown: {combined['maximum_drawdown']:.6g}\n"
        f"- Combined break-even round-trip cost: "
        f"{combined['break_even_roundtrip_bps']:.6g} bps\n"
        f"- Primary conclusion: the statistical signal {conclusion}.\n\n"
        "Gross and net P&L are reported separately. Aggregate trades do not expose "
        "the book, queue position or exact executable spread, so this is a sensitivity "
        "analysis rather than a fill simulator. The final holdout remains sealed.\n",
        encoding="utf-8",
    )

    payload = {
        "phase": 6,
        "config_sha256": sha256_file(Path("configs/phase6.yaml")),
        "protocol_sha256": sha256_file(Path("docs/phase6_protocol.md")),
        "final_model_specification_hash": final_specification["specification_hash"],
        "evaluation_config": evaluation_config.name,
        "thresholds": thresholds,
        "development_volatility": volatility,
        "portfolio_weights": weights,
        "outputs": {name: sha256_file(path) for name, path in outputs.items()},
        "ledgers": {
            "asset": sha256_file(asset_ledger_path),
            "portfolio": sha256_file(portfolio_ledger_path),
        },
        "primary_conclusion": conclusion,
    }
    manifest_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    manifest = {**payload, "manifest_hash": manifest_hash}
    manifest_path = evaluation_config.data.manifest_dir / "phase6-execution.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest_hash": manifest_hash,
        "primary_net_pnl": combined["net_pnl"],
        "primary_trades": combined["trades"],
        "conclusion": conclusion,
    }
