# Spot–Perpetual Price Discovery Research Lab

A reproducible market-microstructure research system for testing whether signed
trading activity in BTC and ETH perpetual futures contains incremental information
about subsequent spot returns.

The project covers verified public-data ingestion, causal fixed-grid features,
walk-forward modelling, untouched confirmation and holdout evaluation, explicit
latency/cost analysis, and a parity-tested C++ replay kernel.

## Results

The frozen BTC model retained predictive power on the untouched ten-day holdout:

| Metric | Result |
|---|---:|
| Final observations | 863,926 |
| Five-second OOS R² | 0.01763 |
| Pearson information coefficient | 0.1334 |
| Rank information coefficient | 0.1982 |
| Replay-kernel speed-up | 81.76× |

Realised returns increased monotonically across prediction deciles. The registered
execution test generated positive gross P&L, but the forecast edge was smaller than
the prespecified transaction-cost assumption. This supports short-horizon predictive
association, not a claim of causality or deployable trading alpha.

![Final predictive and economic results](reports/final/final-results.png)

The complete interpretation, provenance and limitations are in the
[`final research report`](docs/final_report.md).

## What the system does

- Downloads official Binance spot and USD-M futures daily aggregate-trade archives.
- Verifies vendor checksums and records content-addressed raw/Parquet manifests.
- Normalises both source schemas into typed, partitioned Parquet.
- Builds one-second decision rows from right-labelled 100 ms bars.
- Enforces a full-bar feature lag and constructs future labels separately.
- Compares spot-only and expanded spot/perpetual models through time-based splits.
- Runs deterministic robustness checks, placebo alignment tests and ETH replication.
- Simulates first-future-trade execution under explicit latency and cost assumptions.
- Provides equivalent Python and C++20/pybind11 two-market replay implementations.
- Archives the final model, protocol, reports and release artifacts by SHA-256.

## Research design

The sample is divided chronologically:

| Period | Dates | Purpose |
|---|---|---|
| Development | 2025-01-02 to 2025-01-31 | Walk-forward comparison and model tuning |
| Confirmation | 2025-02-01 to 2025-02-20 | Model selection and robustness checks |
| Final holdout | 2025-02-21 to 2025-03-02 | One-time untouched evaluation |

BTCUSDT is the primary asset and the next five-second BTC spot log return is the
primary target. ETHUSDT is a prespecified replication asset. The final model is an
expanded XGBoost regressor using spot, perpetual and cross-market predictors.

The design, model-selection rule and final evaluator were frozen and content-hashed
before their respective evaluation data were accessed. The final-evaluation command
refuses a second run while its result manifest exists.

## Data and leakage controls

- Trades enter the first 100 ms boundary strictly after their event timestamp.
- Every predictor is shifted by one complete base bar before decision sampling.
- Spot and perpetual inputs meet on an exact shared grid; no future as-of join exists.
- Future-return labels are generated separately and excluded from predictors.
- Whole-day expanding splits include a ten-second purge and evaluation embargo.
- Preprocessing is fitted only on the training portion of each split.
- Raw and processed market data are ignored by Git; compact manifests retain their
  checksums, row counts, timestamp units and schema versions.

## Execution assumptions

Signals enter on the first observed spot trade at or after decision time plus the
configured positive latency. Exit occurs on the first trade at least five seconds
after entry, and positions cannot overlap within an asset. Gross and net results are
reported separately across multiple latency and all-in round-trip cost assumptions.

This is a transparent sensitivity analysis over aggregate trades. It cannot reproduce
quotes, available depth, queue priority, market impact or adverse selection.

## C++ replay kernel

The C++20/pybind11 component accelerates the bounded two-market merge and fixed-grid
base aggregation defined by the pure-Python reference. Outputs match exactly for
integer columns and within `1e-12` for floating-point columns.

On the frozen two-million-event equivalent-work benchmark, Python processed 560,400
events/s and C++ processed 45.8 million events/s, an 81.76× kernel speed-up. This is
not an end-to-end pipeline or Polars comparison.

## Installation

Requires Python 3.12+ and a C++20 toolchain.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

With `uv` installed:

```bash
uv sync --extra dev
```

## Usage

The two-day smoke configuration exercises the reproducible data path:

```bash
spot-perp show-config --config configs/smoke.yaml
spot-perp download --config configs/smoke.yaml
spot-perp normalise --config configs/smoke.yaml
spot-perp validate --config configs/smoke.yaml
spot-perp features --config configs/smoke.yaml
```

Research and engineering commands:

```bash
spot-perp train --config configs/development.yaml
spot-perp confirm \
  --development-config configs/development.yaml \
  --confirmation-config configs/confirmation.yaml
spot-perp backtest --help
spot-perp benchmark-replay --events-per-market 1000000 --repeats 3
```

The final holdout has already been evaluated. Verify its archived artifacts without
rerunning it:

```bash
scripts/reproduce_final.sh
```

## Verification

```bash
pytest
ruff format --check .
ruff check .
mypy
```

The release contains 54 automated tests covering configuration, checksums,
normalisation, feature causality, research splits, execution timing, reconciliation,
Python/C++ parity and the one-time holdout guard.

## Documentation

- [`Final research report`](docs/final_report.md)
- [`Methodology`](docs/methodology.md)
- [`Feature dictionary`](docs/feature_dictionary.md)
- [`Data dictionary`](docs/data_dictionary.md)
- [`Research design`](docs/research_design.md)
- [`Limitations`](docs/limitations.md)
- [`Decision and provenance log`](docs/decision_log.md)
- [`Holdout-opening record`](docs/holdout_opening_record.md)

Historical frozen protocols and manifests remain in the repository as an immutable
audit trail. The `v1.0.0` tag identifies the completed research release.
