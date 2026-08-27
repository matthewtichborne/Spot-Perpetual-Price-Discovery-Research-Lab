"""Strict Phase 6 execution configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Phase6Config(BaseModel):
    """Frozen economic-backtest controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    development_config: Path
    evaluation_config: Path
    reports_dir: Path
    ledger_dir: Path
    holding_seconds: int = Field(gt=0)
    signal_threshold_sigma: float = Field(gt=0)
    latencies_ms: tuple[int, ...]
    primary_latency_ms: int = Field(gt=0)
    entry_fee_bps: float = Field(ge=0)
    exit_fee_bps: float = Field(ge=0)
    slippage_bps_per_side: float = Field(ge=0)
    spread_proxy_bps_per_side: float = Field(ge=0)
    cost_sensitivity_roundtrip_bps: tuple[float, ...]
    maximum_gross_exposure: float = Field(gt=0)
    inverse_volatility_minimum_multiplier: float = Field(gt=0)
    inverse_volatility_maximum_multiplier: float = Field(gt=0)
    daily_loss_limit: float = Field(gt=0)
    annualisation_days: int = Field(gt=0)
    random_seed: int

    @model_validator(mode="after")
    def coherent_grids(self) -> Phase6Config:
        if tuple(sorted(set(self.latencies_ms))) != self.latencies_ms:
            raise ValueError("latencies_ms must be sorted and unique")
        if self.primary_latency_ms not in self.latencies_ms:
            raise ValueError("primary_latency_ms must be in latencies_ms")
        if tuple(sorted(set(self.cost_sensitivity_roundtrip_bps))) != (
            self.cost_sensitivity_roundtrip_bps
        ):
            raise ValueError("cost sensitivity grid must be sorted and unique")
        if self.inverse_volatility_minimum_multiplier > self.inverse_volatility_maximum_multiplier:
            raise ValueError("inverse-volatility multiplier bounds are reversed")
        if self.reference_roundtrip_cost_bps not in self.cost_sensitivity_roundtrip_bps:
            raise ValueError("reference all-in cost must be present in sensitivity grid")
        return self

    @property
    def reference_roundtrip_cost_bps(self) -> float:
        """Return the additive all-in reference round-trip cost."""

        return (
            self.entry_fee_bps
            + self.exit_fee_bps
            + 2 * self.slippage_bps_per_side
            + 2 * self.spread_proxy_bps_per_side
        )


def load_phase6_config(path: Path) -> Phase6Config:
    """Load and strictly validate the Phase 6 YAML document."""

    with path.open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return Phase6Config.model_validate(raw)
