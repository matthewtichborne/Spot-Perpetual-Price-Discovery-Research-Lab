# Spot-Perpetual Price Discovery Research Lab

This project investigates whether signed trading activity in BTC and ETH USD-M
perpetual futures adds out-of-sample information about subsequent spot returns after
controlling for spot-market information. It will then test whether any relationship
survives explicit latency and transaction-cost assumptions. No research result has
been claimed yet.

## Current status

Phase 0 is complete: Binance's official public archives are accessible without
credentials, checksums are available, BTC/ETH spot and perpetual data overlap, and
the source formats are feasible. Phase 1 provides a typed configuration and CLI
scaffold with deterministic tests. See [`docs/feasibility.md`](docs/feasibility.md)
for evidence and limitations.

## Quick start

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
spot-perp show-config --config configs/smoke.yaml
pytest
ruff check .
mypy
```

With `uv` installed, `uv sync --extra dev` and `uv run ...` can be used instead.

## Research safeguards

- Time splits only; observations are never randomly shuffled.
- The final holdout remains sealed until the research design is frozen.
- Spot aggregate trades from 2025 onward use microsecond timestamps, while the
  sampled USD-M futures archive uses milliseconds.
- “Flow imbalance” means signed aggregate-trade flow, not order-book imbalance.
- Gross and net results will be reported separately; no performance metrics exist yet.

## Planned workflow

```text
official ZIP + checksum -> validated raw archive -> normalised Parquet
-> leakage-safe features -> walk-forward models -> latency/cost backtest -> report
```

The detailed source of truth is
[`Spot_Perpetual_Project_Implementation_Plan.md`](Spot_Perpetual_Project_Implementation_Plan.md).
