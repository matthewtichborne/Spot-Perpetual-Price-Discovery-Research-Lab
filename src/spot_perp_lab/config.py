"""Validated experiment configuration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class DateRange(BaseModel):
    """Inclusive UTC date range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date

    @model_validator(mode="after")
    def ordered(self) -> DateRange:
        if self.end < self.start:
            raise ValueError("date range end must not precede start")
        return self


class DataConfig(BaseModel):
    """Public archive selection and local storage settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: HttpUrl = HttpUrl("https://data.binance.vision")
    symbols: tuple[Literal["BTCUSDT", "ETHUSDT"], ...]
    markets: tuple[Literal["spot", "perpetual"], ...] = ("spot", "perpetual")
    dates: DateRange
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    manifest_dir: Path = Path("data/manifests")


class ResearchConfig(BaseModel):
    """Leakage-sensitive research controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_horizon_seconds: int = Field(default=5, gt=0)
    holdout_status: Literal["not_applicable", "sealed", "open"] = "sealed"
    design_status: Literal["preliminary", "frozen"] = "preliminary"
    primary_symbol: Literal["BTCUSDT", "ETHUSDT"] = "BTCUSDT"
    initial_train_days: int = Field(default=10, gt=0)
    test_days: int = Field(default=5, gt=0)
    purge_seconds: int = Field(default=10, ge=0)
    bootstrap_replicates: int = Field(default=2_000, gt=0)
    ridge_alpha: float = Field(default=1.0, gt=0)
    logistic_c: float = Field(default=1.0, gt=0)
    hac_max_lags: int = Field(default=12, ge=0)
    reports_dir: Path = Path("reports/development")
    random_seed: int = 20260825


class FeatureConfig(BaseModel):
    """Causal fixed-grid feature and label settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_interval_ms: int = Field(default=100, gt=0)
    decision_interval_ms: int = Field(default=1_000, gt=0)
    windows_ms: tuple[int, ...] = (100, 500, 1_000, 5_000, 10_000)
    label_horizons_ms: tuple[int, ...] = (1_000, 5_000, 10_000)
    basis_z_window_ms: int = Field(default=10_000, gt=0)
    feature_lag_bars: int = Field(default=1, ge=1)
    output_dir: Path = Path("data/processed/features")

    @model_validator(mode="after")
    def aligned_intervals(self) -> FeatureConfig:
        intervals = (
            self.decision_interval_ms,
            self.basis_z_window_ms,
            *self.windows_ms,
            *self.label_horizons_ms,
        )
        if any(interval % self.base_interval_ms for interval in intervals):
            raise ValueError("all feature intervals must be multiples of base_interval_ms")
        if tuple(sorted(set(self.windows_ms))) != self.windows_ms:
            raise ValueError("windows_ms must be sorted and unique")
        if tuple(sorted(set(self.label_horizons_ms))) != self.label_horizons_ms:
            raise ValueError("label_horizons_ms must be sorted and unique")
        return self


class AppConfig(BaseModel):
    """Root application configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    data: DataConfig
    research: ResearchConfig = ResearchConfig()
    features: FeatureConfig = FeatureConfig()


def load_config(path: Path) -> AppConfig:
    """Load and strictly validate a YAML configuration file."""

    with path.open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return AppConfig.model_validate(raw)
