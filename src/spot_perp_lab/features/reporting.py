"""Deterministic descriptive artifacts for Phase 3 feature data."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib
import polars as pl

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def write_descriptive_artifacts(
    paths: list[Path],
    config_name: str,
    predictors: list[str],
    labels: list[str],
) -> tuple[Path, Path]:
    """Write a compact summary table and a smoke diagnostic figure."""

    table_path = Path("reports/tables") / f"{config_name}-phase3-summary.csv"
    figure_path = Path("reports/figures") / f"{config_name}-phase3-overview.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, int | float | str]] = []
    plot_samples: list[pl.DataFrame] = []
    plot_symbol: str | None = None
    for path in paths:
        frame = pl.read_parquet(path, hive_partitioning=False)
        frame_symbol = str(frame["symbol"][0])
        if plot_symbol is None:
            plot_symbol = frame_symbol
        primary_label = "target_spot_log_return_5000ms"
        summary_rows.append(
            {
                "symbol": frame_symbol,
                "date": str(frame["date"][0]),
                "rows": frame.height,
                "predictor_count": len(predictors),
                "label_count": len(labels),
                "first_decision_time_ns": int(frame["decision_time_ns"][0]),
                "last_decision_time_ns": int(frame["decision_time_ns"][-1]),
                "primary_label_nulls": frame[primary_label].null_count(),
                "mean_spot_trades_1s": float(
                    cast(float | int | None, frame["spot_trade_count_1000ms"].mean()) or 0
                ),
                "mean_perpetual_trades_1s": float(
                    cast(float | int | None, frame["perpetual_trade_count_1000ms"].mean()) or 0
                ),
                "mean_log_basis": float(
                    cast(float | int | None, frame["spot_perp_log_basis"].mean()) or 0
                ),
                "std_log_basis": float(
                    cast(float | int | None, frame["spot_perp_log_basis"].std()) or 0
                ),
                "std_target_spot_return_5s": float(
                    cast(float | int | None, frame[primary_label].std()) or 0
                ),
            }
        )
        if frame_symbol == plot_symbol:
            plot_samples.append(
                frame.select(
                    "decision_time_ns",
                    "spot_perp_log_basis",
                    "spot_quantity_imbalance_1000ms",
                    "perpetual_quantity_imbalance_1000ms",
                ).with_columns(
                    pl.col("spot_quantity_imbalance_1000ms")
                    .rolling_mean(60, min_samples=1)
                    .alias("spot_imbalance_plot"),
                    pl.col("perpetual_quantity_imbalance_1000ms")
                    .rolling_mean(60, min_samples=1)
                    .alias("perpetual_imbalance_plot"),
                )[::60]
            )
    pl.DataFrame(summary_rows).sort(["date", "symbol"]).write_csv(table_path)

    sample = pl.concat(plot_samples).sort("decision_time_ns")
    start = int(sample["decision_time_ns"][0])
    elapsed_hours = (sample["decision_time_ns"].to_numpy() - start) / 3_600_000_000_000
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(elapsed_hours, sample["spot_perp_log_basis"].to_numpy(), linewidth=0.8)
    axes[0].set_ylabel("log basis")
    axes[0].set_title(f"{plot_symbol} spot-perpetual Phase 3 diagnostics")
    axes[1].plot(
        elapsed_hours,
        sample["spot_imbalance_plot"].to_numpy(),
        label="spot",
        linewidth=0.7,
        alpha=0.8,
    )
    axes[1].plot(
        elapsed_hours,
        sample["perpetual_imbalance_plot"].to_numpy(),
        label="perpetual",
        linewidth=0.7,
        alpha=0.8,
    )
    axes[1].set_xlabel("hours from first decision")
    axes[1].set_ylabel("60 s mean of 1 s imbalance")
    axes[1].legend(loc="upper right")
    figure.savefig(figure_path, dpi=140, metadata={"Software": "spot-perp-lab"})
    plt.close(figure)
    return table_path, figure_path
