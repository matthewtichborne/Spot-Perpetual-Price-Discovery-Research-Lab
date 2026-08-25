# Spot-Perpetual Price Discovery Research Lab

## Copy-paste prompt for a new Codex chat

I want to build a portfolio-quality quantitative research project called **Spot-Perpetual Price Discovery Research Lab**. The project should investigate whether signed trading activity in BTC and ETH perpetual futures contains incremental information about subsequent spot-market returns, and whether any predictive relationship survives execution delays and transaction costs.

The project is intended for graduate quantitative research, trading, analyst and developer applications. It must demonstrate an end-to-end research process, large-market-data engineering, statistical inference, machine learning, realistic backtesting, risk analysis and production-quality software practices.

Use the detailed implementation plan in `Spot_Perpetual_Project_Implementation_Plan.md` as the source of truth. Work incrementally. Begin with the Phase 0 feasibility checks and Phase 1 repository scaffold. Do not invent results or CV metrics. Do not open the final holdout period until the research design and model-selection procedure have been frozen. Prefer simple, interpretable baselines before nonlinear models. Add the C++ component only after the Python reference implementation is correct and tested.

At every milestone:

1. Implement the smallest complete slice.
2. Add or update tests.
3. Run the relevant checks.
4. Summarise what changed and what remains.
5. Record any methodological decision in `docs/decision_log.md`.

The first task is to verify that the official Binance public spot and USD-M futures aggregate-trade files are accessible, inspect their current schemas and timestamp units, and produce a short feasibility report. If the source is unavailable, stop and propose a compliant public-data fallback rather than silently changing the research question.

---

## 1. Project objective

Build a reproducible research and execution pipeline answering:

> Does signed trade flow in BTC and ETH perpetual futures improve forecasts of subsequent spot returns beyond information already present in the spot market, and does any improvement remain economically meaningful after latency and transaction costs?

The finished repository should show four capabilities:

1. **Research:** hypothesis formation, statistical testing, predictive modelling and honest interpretation.
2. **Data engineering:** ingestion, validation, timestamp normalisation, columnar storage and analytical SQL.
3. **Trading analysis:** event-driven execution, fees, slippage, latency, positions and portfolio risk.
4. **Software engineering:** modular design, tests, continuous integration, reproducible environments and a bounded C++ optimisation.

This project should eventually replace the inaccessible Directional-Change Trading project on the CV rather than sit beside another unsupported market-data project.

## 2. Definition of done

The project is CV-ready only when all of the following are true:

- The complete workflow can be reproduced from documented commands.
- Data files are downloaded from a public source and verified against checksums where supplied.
- Raw and normalised schemas are documented.
- Every timestamp is normalised and tested without relying on guessed units.
- The primary hypothesis, features, horizons and holdout period are frozen before final evaluation.
- At least one simple baseline, one regularised statistical model and one nonlinear model are compared.
- All reported results are genuinely out of sample.
- Transaction costs, slippage and latency are applied explicitly.
- Gross and net performance are reported separately.
- Results include uncertainty estimates and robustness checks.
- A deterministic small-data workflow runs in GitHub Actions.
- The Python test suite passes and the C++ implementation agrees with the Python reference.
- The repository contains a polished README, methodology, results, limitations and reproduction instructions.
- Every number used on the CV can be regenerated from a versioned report artifact.

## 3. Scope

### Primary scope

- Exchange: Binance public market data.
- Markets:
  - BTCUSDT spot
  - BTCUSDT USD-M perpetual future
  - ETHUSDT spot
  - ETHUSDT USD-M perpetual future
- Source type: aggregate trades (`aggTrades`).
- Initial smoke-test period: two consecutive days of BTC spot and perpetual data.
- Development sample: approximately 30 consecutive days.
- Final research sample: target 60-90 consecutive days, subject to storage and download feasibility.
- Primary forecast target: next five-second BTC spot log return.
- Secondary targets: one-second and ten-second returns; ETH replication.

### Non-goals

- No live trading or exchange order submission.
- No claim that a backtest represents executable institutional capacity.
- No reinforcement learning in the first version.
- No deep neural network unless simple models establish a credible signal and there is a clear research reason.
- No Spark, Kubernetes, cloud cluster or other infrastructure added only for keyword coverage.
- No random train/test splits.
- No parameter selection using the final holdout period.
- No performance numbers chosen retrospectively because they look impressive.

## 4. Planned toolkit

Use each tool for a clear reason:

- **Python 3.12+:** orchestration, modelling, reporting and reference implementations.
- **Polars:** fast columnar transformations and time aggregation.
- **DuckDB/SQL:** analytical queries over Parquet datasets.
- **Apache Parquet/PyArrow:** partitioned columnar storage and schema interoperability.
- **NumPy:** numerical arrays and Python/C++ interfaces.
- **scikit-learn:** preprocessing, baselines, regularised models and evaluation.
- **XGBoost:** nonlinear predictive-model comparison after baselines.
- **statsmodels:** HAC/Newey-West inference, VAR or Granger analysis where appropriate.
- **C++20, CMake and pybind11:** event-stream merge or replay hot path with Python bindings.
- **pytest:** unit, integration, regression and Python/C++ parity tests.
- **Ruff and mypy:** formatting, linting and type checking.
- **Docker:** reproducible build and execution environment.
- **GitHub Actions:** small-data CI workflow.
- **Typer:** command-line interface.
- **Pydantic:** validated experiment configuration if configuration complexity justifies it.

Avoid adding MLflow, DVC or a cloud deployment until the core research is complete. A simple immutable run manifest is enough initially.

## 5. Repository structure

```text
spot-perp-research/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── CMakeLists.txt
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── smoke.yaml
│   ├── development.yaml
│   └── final.yaml
├── cpp/
│   ├── include/
│   ├── src/
│   ├── bindings/
│   ├── tests/
│   └── benchmarks/
├── data/
│   ├── raw/              # ignored by Git
│   ├── interim/          # ignored by Git
│   ├── processed/        # ignored by Git
│   ├── manifests/        # committed when small
│   └── fixtures/         # deterministic synthetic CI data
├── docs/
│   ├── research_design.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── limitations.md
│   └── decision_log.md
├── notebooks/
│   └── exploratory/      # exploration only; production logic stays in src
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── development/
│   └── final/
├── scripts/
│   └── reproduce.sh
├── src/
│   └── spot_perp_lab/
│       ├── cli.py
│       ├── config.py
│       ├── logging.py
│       ├── data/
│       │   ├── download.py
│       │   ├── checksums.py
│       │   ├── schemas.py
│       │   ├── normalise.py
│       │   ├── validate.py
│       │   └── manifest.py
│       ├── features/
│       │   ├── bars.py
│       │   ├── trade_flow.py
│       │   ├── basis.py
│       │   ├── volatility.py
│       │   └── labels.py
│       ├── research/
│       │   ├── splits.py
│       │   ├── baselines.py
│       │   ├── models.py
│       │   ├── inference.py
│       │   ├── placebo.py
│       │   └── evaluation.py
│       ├── backtest/
│       │   ├── engine.py
│       │   ├── execution.py
│       │   ├── costs.py
│       │   ├── portfolio.py
│       │   ├── risk.py
│       │   └── metrics.py
│       └── reporting/
│           ├── figures.py
│           ├── tables.py
│           └── report.py
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    └── parity/
```

## 6. Canonical normalised trade schema

The normalised event table should contain at least:

| Field | Type | Meaning |
|---|---:|---|
| `exchange` | string | Data source, initially `binance` |
| `market_type` | categorical | `spot` or `perpetual` |
| `symbol` | categorical | `BTCUSDT` or `ETHUSDT` |
| `event_time_ns` | int64 | UTC event timestamp normalised to nanoseconds |
| `aggregate_trade_id` | int64 | Source aggregate-trade identifier |
| `first_trade_id` | int64 | First underlying trade identifier if available |
| `last_trade_id` | int64 | Last underlying trade identifier if available |
| `price` | float64 | Trade price |
| `quantity` | float64 | Base-asset quantity |
| `notional` | float64 | `price * quantity` |
| `is_buyer_maker` | bool | Source aggressor-side flag |
| `aggressor_sign` | int8 | +1 for buyer-initiated, -1 for seller-initiated |
| `signed_quantity` | float64 | `aggressor_sign * quantity` |
| `signed_notional` | float64 | `aggressor_sign * notional` |
| `source_file` | string | Original archive name |

Important: aggregate-trade imbalance is **signed trade-flow imbalance**, not limit-order-book order-flow imbalance. Documentation and CV wording must preserve that distinction.

## 7. Research design

### Primary hypothesis

After controlling for lagged spot returns and spot signed trade flow, perpetual-futures signed trade flow provides incremental information about the next five-second spot return.

### Primary comparison

- Baseline model: lagged spot returns, spot signed trade flow, spot volume and realised volatility.
- Expanded model: baseline features plus perpetual signed trade flow, perpetual intensity, spot-perpetual basis and cross-market relative activity.
- Primary test: difference in genuine out-of-sample performance between expanded and baseline models.

### Feature families

For windows such as 100 ms, 500 ms, one second, five seconds and ten seconds:

- Signed quantity imbalance
- Signed notional imbalance
- Buyer- and seller-initiated trade counts
- Total quantity and notional
- Trade-arrival intensity
- Average trade size
- Spot and perpetual lagged log returns
- Spot-perpetual log basis
- Basis change and rolling basis z-score
- Realised volatility
- Relative spot/perpetual volume and intensity
- Interaction between flow imbalance and volatility regime

All rolling features must use data available at or before decision time. Any feature aligned to a fixed time grid must be lagged explicitly before prediction.

### Targets

- Continuous future spot log return over one, five and ten seconds.
- Directional label as a secondary target.
- Optional economic label: future return exceeding an explicitly configured cost threshold.

### Models

Build in this order:

1. Training-mean or no-change baseline.
2. Same-market autoregressive/spot-flow baseline.
3. Linear regression with HAC inference.
4. Logistic regression for direction.
5. Elastic Net or Ridge/Lasso.
6. XGBoost with a deliberately small search space.
7. Optional VAR/Granger analysis for interpretation, not as proof of tradable causality.

### Metrics

For regression:

- Out-of-sample R-squared
- Mean squared error and mean absolute error
- Pearson and rank information coefficient
- Calibration by prediction decile

For classification:

- ROC AUC
- Precision-recall AUC
- Brier score
- Accuracy only alongside class balance
- Return or hit rate by prediction decile

For incremental value:

- Expanded model minus baseline metric
- Confidence interval for the difference
- Performance by day and regime
- Signal decay across horizons

## 8. Data splitting and leakage controls

- Split only by time and preferably by whole UTC day.
- Never shuffle observations.
- Reserve the final 15-20% of dates as a sealed holdout.
- Use expanding or rolling walk-forward folds on the development period.
- Purge at least the maximum label horizon around fold boundaries.
- Apply an embargo if overlapping rolling features can leak information across boundaries.
- Fit scalers and feature selection on training data only.
- Select XGBoost hyperparameters using the validation portion only.
- Freeze `docs/research_design.md` and the final config before evaluating the holdout.
- Record the commit hash and data-manifest hash used to open the holdout.

Recommended initial layout for 60 days:

- Days 1-40: rolling training and validation development.
- Days 41-50: model-selection confirmation.
- Days 51-60: untouched final holdout.

Adjust only before viewing holdout results.

## 9. Statistical inference and robustness

Required checks:

- Newey-West/HAC standard errors for autocorrelated return regressions.
- Day-blocked bootstrap confidence intervals for model and strategy metrics.
- Baseline-versus-expanded paired comparison by day.
- Placebo test with deliberately time-shifted perpetual signals.
- Sign-flip or permutation test performed at a block level where appropriate.
- Separate high- and low-volatility regimes.
- Separate BTC development and ETH replication.
- Stability across horizons and feature windows.
- Sensitivity to winsorisation and extreme-event handling.
- Sensitivity to missing-data rules.
- Multiple-testing disclosure; use a correction if many hypotheses are promoted as findings.

The report must distinguish statistical predictability from economic tradability.

## 10. Backtest specification

### Event timing

- Generate a signal at decision time `t` using only information timestamped at or before `t`.
- Apply a configured latency.
- Execute using the first eligible observed spot trade at or after `t + latency`.
- Never execute at the price used to create the signal.

### Cost model

Use configuration rather than hard-coded current exchange fees:

- Entry fee in basis points
- Exit fee in basis points
- Additional slippage in basis points
- Optional conservative spread proxy
- Cost-sensitivity grid

Because aggregate trades do not provide the complete order book, describe this as a cost-sensitivity model rather than an exact fill simulator.

### Position and risk rules

- Positions: short, flat or long initially.
- No overlapping positions in the first reference backtest.
- Fixed maximum gross exposure.
- Configurable holding horizon or exit condition.
- Position sizing initially constant; inverse-volatility sizing as a secondary experiment.
- Daily loss and position limits in the portfolio version.
- BTC and ETH portfolio weights normalised to the risk budget.

### Latency scenarios

Evaluate a pre-specified grid such as:

- 0.1 seconds
- 0.5 seconds
- 1 second
- 2 seconds
- 5 seconds

The exact grid should reflect timestamp resolution and be frozen before the holdout.

### Trading metrics

- Number of trades
- Exposure and turnover
- Gross and net P&L
- Daily return series
- Annualised Sharpe ratio calculated from daily returns
- Sortino ratio as secondary
- Maximum drawdown
- Win rate and average win/loss
- Average holding period
- Cost break-even point
- P&L concentration by day
- Performance by volatility regime
- Performance degradation with latency
- BTC, ETH and combined portfolio results

Do not annualise trade-level returns as though they were independent observations.

## 11. C++ component

Implement only after the Python reference pipeline is correct.

Recommended bounded component:

- Merge time-ordered spot and perpetual event streams.
- Maintain rolling signed-flow and volume aggregates.
- Emit fixed-grid feature rows or as-of alignment indices.
- Expose the implementation to Python through pybind11.

Requirements:

- Pure Python reference implementation.
- Identical results within explicit numerical tolerances.
- Unit tests for empty arrays, duplicates, ties and out-of-order input.
- Benchmark on a fixed event sample.
- Report events per second, runtime and speed-up.
- Do not claim a speed-up against Polars unless the benchmark actually compares equivalent work.

## 12. Test plan

### Data tests

- Correct URL and filename construction.
- Checksum verification.
- Schema detection for spot and futures files.
- Millisecond versus microsecond timestamp detection.
- Strictly non-decreasing event time after sorting.
- Duplicate identifier handling.
- Correct buyer/seller aggressor sign.
- Price, quantity and notional sanity checks.
- Partition and manifest reproducibility.

### Feature tests

- Hand-calculated signed-flow examples.
- Rolling-window boundary behaviour.
- Spot-perpetual as-of alignment never uses future data.
- Basis and return calculations.
- Label horizons and final-row null handling.
- Feature lagging.

### Split tests

- Chronological ordering.
- No date overlap.
- Purge and embargo sizes.
- Scaler fitted only on training data.
- Holdout dates inaccessible to model selection.

### Backtest tests

- Execution occurs after decision time plus latency.
- Fees and slippage have correct signs.
- Position transitions and cash accounting.
- No overlapping position when prohibited.
- Drawdown and daily return calculations.
- Deterministic results from a fixed fixture.

### C++ tests

- Python/C++ parity.
- Timestamp ties.
- Empty and single-event inputs.
- Multiple events inside one bucket.
- Large-array benchmark does not alter results.

## 13. Reproducibility and experiment records

Every run should write a manifest containing:

- Run identifier
- UTC run timestamp
- Git commit hash
- Configuration hash
- Raw-data file names and checksums
- Processed-data schema version
- Training, validation and test dates
- Feature list
- Model type and parameters
- Random seeds
- Package versions
- Output table and figure paths

The command below should eventually reproduce the public result from downloaded data:

```bash
./scripts/reproduce.sh configs/final.yaml
```

Recommended CLI:

```bash
uv run spot-perp download --config configs/smoke.yaml
uv run spot-perp normalise --config configs/smoke.yaml
uv run spot-perp validate --config configs/smoke.yaml
uv run spot-perp features --config configs/development.yaml
uv run spot-perp train --config configs/development.yaml
uv run spot-perp backtest --config configs/development.yaml
uv run spot-perp report --config configs/development.yaml
```

## 14. Milestones and acceptance criteria

### Phase 0: feasibility spike

Tasks:

- Verify official data access without credentials.
- Download one spot and one futures aggregate-trade file.
- Inspect archive names, columns, timestamp units, checksums and file sizes.
- Estimate storage for 30, 60 and 90 days.
- Confirm that BTC and ETH files overlap for the desired period.
- Write `docs/feasibility.md`.

Acceptance criteria:

- Two source files are downloaded and parsed.
- Schemas and timestamp units are recorded from evidence.
- Storage estimate and fallback plan are documented.
- A clear go/no-go decision is made before building the full pipeline.

### Phase 1: repository and quality scaffold

Tasks:

- Initialise Git repository and Python package.
- Configure `uv`, Ruff, mypy and pytest.
- Add CLI skeleton and typed configuration.
- Add Dockerfile and initial CI workflow.
- Add deterministic synthetic fixtures.

Acceptance criteria:

- Fresh clone can install dependencies.
- `pytest`, Ruff and mypy pass.
- Docker image builds.
- CI passes without downloading the full dataset.

### Phase 2: ingestion and normalisation

Tasks:

- Implement downloader, checksum validation and caching.
- Parse spot and futures formats explicitly.
- Normalise timestamps and aggressor side.
- Write partitioned Parquet by symbol, market type and date.
- Build data manifest and DuckDB validation queries.

Acceptance criteria:

- Two-day BTC smoke dataset builds end to end.
- Re-running is idempotent.
- Manifest hashes are stable.
- Validation catches deliberately corrupted fixtures.
- Raw CSV extraction does not need to remain on disk after verified Parquet creation.

### Phase 3: features and labels

Tasks:

- Create fixed-grid bars and cross-market alignment.
- Implement primary feature families.
- Implement future-return labels.
- Add data-quality and leakage tests.
- Produce descriptive tables and plots.

Acceptance criteria:

- Every feature has a definition in the data dictionary.
- Hand-worked examples match tests.
- No feature uses data after decision time.
- Feature generation is deterministic.

### Phase 4: research baselines

Tasks:

- Freeze primary hypothesis and development splits.
- Implement baseline and expanded linear models.
- Add regularised and logistic models.
- Add HAC and blocked-bootstrap inference.
- Produce walk-forward results by fold and day.

Acceptance criteria:

- Baseline and expanded models are directly comparable.
- All preprocessing occurs inside training folds.
- Results include uncertainty, not only point estimates.
- Model failures and null results are retained in reports.

### Phase 5: nonlinear model and robustness

Tasks:

- Add XGBoost with limited validation-only tuning.
- Run placebo, regime, horizon and ETH-replication checks.
- Freeze the final model-selection rule.
- Write the holdout-opening record.

Acceptance criteria:

- Hyperparameter search space is documented.
- Nonlinear model is compared against simple models, not presented alone.
- Research design is frozen before holdout access.

### Phase 6: execution and portfolio analysis

Tasks:

- Implement event-timed execution.
- Add configurable fees, slippage and latency.
- Add risk and position rules.
- Add BTC, ETH and combined portfolio reporting.
- Run cost and latency sensitivity.

Acceptance criteria:

- Backtest timing tests prove no same-event execution.
- Gross and net results are separated.
- Daily P&L reconciles to trades.
- Performance is reported across costs and latency, not at one favourable assumption.

### Phase 7: C++ replay component

Tasks:

- Profile the Python workflow.
- Implement the bounded merge or rolling-aggregation component.
- Add pybind11 bindings, CMake build, parity tests and benchmark.

Acceptance criteria:

- Python and C++ outputs agree.
- Benchmark method is reproducible.
- Speed-up and throughput are reported honestly.
- CI builds and tests the C++ extension.

### Phase 8: final holdout and publication

Tasks:

- Evaluate the frozen pipeline once on the final holdout.
- Generate final tables and figures.
- Write methodology, findings, robustness and limitations.
- Polish README and reproduction instructions.
- Tag a release and archive the exact result manifest.
- Draft final CV bullets only from generated metrics.

Acceptance criteria:

- Final report can be regenerated from the release tag.
- README explains the question and findings before implementation details.
- Limitations include data, fill modelling, costs, market specificity and multiple testing.
- Repository contains no secrets or large raw-data files.

## 15. GitHub presentation requirements

The README should be understandable in this order:

1. One-paragraph research question and conclusion.
2. A compact results table.
3. One strong figure showing incremental predictive value or signal decay.
4. One cost/latency robustness figure.
5. Data and methodology summary.
6. Architecture diagram or pipeline description.
7. Reproduction commands.
8. Test and CI status.
9. Limitations.
10. Repository structure.

Avoid leading with installation instructions or a large list of technologies. Recruiters should understand the research contribution within one minute.

## 16. Metrics to capture for future CV bullets

### Engineering scale

- Total events
- Raw and processed gigabytes
- Number of days and markets
- Compression ratio
- Pipeline runtime
- DuckDB query time
- C++ events per second
- C++ speed-up
- Test count and optional coverage

### Research

- Number of features and models
- Number of walk-forward folds
- Out-of-sample observations
- Baseline and expanded out-of-sample R-squared
- Information coefficient
- AUC and Brier score
- Confidence intervals and p-values
- Signal-decay horizon
- Robustness across days, regimes and assets

### Trading

- Number of simulated trades
- Gross and net Sharpe ratio
- Maximum drawdown
- Turnover
- Cost break-even point
- Performance at each latency
- Performance versus spot-only baseline
- Portfolio diversification effect

## 17. Provisional CV entry

Do not use this entry until the bracketed values are produced by the final report.

**Spot-Perpetual Price Discovery and Algorithmic Trading**
*Python | C++20/pybind11 | Polars | DuckDB/SQL | Apache Parquet | XGBoost | statsmodels | Docker*

- Engineered a reproducible market-data pipeline processing **[N million] timestamped trades ([X] GB) across [D] days and four BTC/ETH spot and perpetual markets**, using checksum validation, event-time alignment and partitioned Parquet storage; implemented the replay engine in C++20 to process **[N] events per second**.
- Conducted original alpha-signal research across **[F] trade-flow, basis, liquidity and volatility features**, comparing **[M] statistical and machine-learning models over [K] purged walk-forward folds**; achieved an out-of-sample **[AUC/IC/R-squared] of [X] versus [Y] for the baseline**, with signal decay and statistical significance assessed through blocked-bootstrap confidence intervals.
- Built an event-driven algorithmic-trading backtest and BTC/ETH portfolio analysis incorporating fees, slippage, position limits and **[L] latency scenarios**; evaluated **[T] out-of-sample trades**, achieving a net Sharpe ratio of **[X]**, maximum drawdown of **[Y]%** and cost break-even point of **[Z] basis points**.

If net performance is negative, replace the final clause with an honest quantified finding about cost sensitivity or signal decay rather than hiding the result.

## 18. Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| Public files unavailable or regional access differs | Test access in Phase 0; use a documented public fallback only after an explicit decision |
| Timestamp units differ between products or dates | Infer from documented schema and magnitude, normalise explicitly and test boundary dates |
| Dataset is too large | Start with two days, process one archive at a time, write Parquet and remove extracted CSV after verification |
| Apparent lead-lag is timestamp artefact | Use placebo shifts, inspect clock resolution, report limitations and test coarser horizons |
| Leakage through alignment or rolling windows | Use as-of tests, explicit lags, chronological folds, purge and embargo |
| Overfitting from many horizons and features | Pre-register primary hypothesis, restrict tuning and seal a final holdout |
| Backtest overstates executable performance | Execute at the next event, apply cost/latency grids and avoid claims of exact fills |
| Signal is statistically significant but uneconomic | Report both findings; economic failure is a valid research conclusion |
| Crypto appears speculative | Frame the work as exchange microstructure and transferable research methodology |
| C++ expansion delays completion | Keep it to one bounded component after the Python pipeline is complete |
| Repository becomes notebook-driven | Keep production logic under `src`; notebooks call library functions only |

## 19. Suggested initial issue backlog

1. Verify Binance spot and futures aggregate-trade data access.
2. Document source schemas and timestamp units.
3. Estimate data volume for the proposed sample.
4. Scaffold Python package and CI.
5. Create deterministic synthetic fixtures.
6. Implement checksum-aware downloader.
7. Implement spot parser.
8. Implement futures parser.
9. Implement timestamp and aggressor-side normalisation.
10. Write partitioned Parquet and run DuckDB validation.
11. Implement fixed-grid aggregation.
12. Implement signed trade-flow features.
13. Implement basis and volatility features.
14. Implement leakage-safe labels.
15. Implement chronological purged folds.
16. Fit spot-only baseline.
17. Fit cross-market linear model.
18. Add HAC and blocked-bootstrap inference.
19. Add XGBoost comparison.
20. Run placebo and ETH replication.
21. Implement event-driven execution and costs.
22. Add latency and portfolio analysis.
23. Profile and implement C++ replay component.
24. Freeze research design and open holdout.
25. Generate final report, README and CV metrics.

## 20. Rules for the implementation chat

- Never fabricate a metric, result, citation or data property.
- Treat raw data and existing user work as immutable.
- Explain any research-design change before applying it.
- Preserve a clear distinction between exploratory and confirmatory analysis.
- Prefer a correct simple implementation to a complex unfinished one.
- Do not optimise before profiling.
- Do not add a library unless its role is explicit.
- Keep all final results reproducible from configuration and manifests.
- Flag data or methodological limitations prominently.
- Stop before opening the final holdout and ask for confirmation that the design is frozen.
