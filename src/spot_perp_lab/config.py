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


class ResearchConfig(BaseModel):
    """Leakage-sensitive research controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_horizon_seconds: int = Field(default=5, gt=0)
    holdout_status: Literal["not_applicable", "sealed", "open"] = "sealed"
    random_seed: int = 20260825


class AppConfig(BaseModel):
    """Root application configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    data: DataConfig
    research: ResearchConfig = ResearchConfig()


def load_config(path: Path) -> AppConfig:
    """Load and strictly validate a YAML configuration file."""

    with path.open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return AppConfig.model_validate(raw)
