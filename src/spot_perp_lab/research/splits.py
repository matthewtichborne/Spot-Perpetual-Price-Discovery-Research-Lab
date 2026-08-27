"""Whole-day expanding walk-forward splits with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_dates: tuple[str, ...]
    evaluation_dates: tuple[str, ...]


def expanding_day_folds(
    dates: tuple[str, ...], initial_train_days: int, evaluation_days: int
) -> list[WalkForwardFold]:
    """Construct deterministic expanding folds from sorted unique UTC dates."""

    if tuple(sorted(set(dates))) != dates:
        raise ValueError("dates must be sorted and unique")
    if initial_train_days <= 0 or evaluation_days <= 0:
        raise ValueError("fold lengths must be positive")
    folds: list[WalkForwardFold] = []
    train_end = initial_train_days
    while train_end + evaluation_days <= len(dates):
        folds.append(
            WalkForwardFold(
                fold=len(folds) + 1,
                train_dates=dates[:train_end],
                evaluation_dates=dates[train_end : train_end + evaluation_days],
            )
        )
        train_end += evaluation_days
    if not folds:
        raise ValueError("date range is too short for one walk-forward fold")
    return folds


def split_frame(
    frame: pl.DataFrame, fold: WalkForwardFold, purge_seconds: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply whole-day selection plus a boundary purge and embargo."""

    boundary = datetime.fromisoformat(fold.evaluation_dates[0]).replace(tzinfo=UTC)
    boundary_ns = int(boundary.timestamp()) * 1_000_000_000
    purge_ns = purge_seconds * 1_000_000_000
    train = frame.filter(
        pl.col("date").is_in(fold.train_dates)
        & (pl.col("decision_time_ns") < boundary_ns - purge_ns)
    )
    evaluation = frame.filter(
        pl.col("date").is_in(fold.evaluation_dates)
        & (pl.col("decision_time_ns") >= boundary_ns + purge_ns)
    )
    if set(train["date"].unique()) & set(evaluation["date"].unique()):
        raise ValueError("training and evaluation dates overlap")
    if train.height == 0 or evaluation.height == 0:
        raise ValueError(f"fold {fold.fold} is empty after purge/embargo")
    return train, evaluation
