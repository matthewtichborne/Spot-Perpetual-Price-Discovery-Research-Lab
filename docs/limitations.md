# Limitations

## Data and market coverage

- Aggregate trades do not expose the full order book, quoted spread, queue position
  or available depth.
- Signed aggregate-trade flow is not limit-order-book order-flow imbalance.
- Binance-specific findings may not generalise across venues or market regimes.
- The final holdout contains ten contiguous days on one venue and one primary asset.
  It is untouched evidence, but not broad temporal or cross-venue validation.
- ETH replication supplies cross-asset evidence, not cross-venue or long-horizon
  evidence.
- Archive timing, clock alignment and exchange-specific microstructure can differ
  from information available to an executable real-time system.

## Statistical interpretation

- Five-second labels overlap at one-second decision frequency. Fold/day reporting,
  day-block bootstrap intervals and HAC errors address dependence only approximately.
- The development analysis compares several prespecified model/scope combinations;
  its preferred specification is a selection result rather than an untouched estimate.
- The confirmation period is later than development but remains one contiguous
  twenty-day market regime.
- The circular-shift placebo is one negative control and cannot eliminate every clock,
  feed-construction or common-information explanation.
- Stronger one-second than ten-second metrics may reflect short-lived common reactions
  as well as perpetual-to-spot price discovery.
- OOS R² and information coefficients establish forecast association, not causality
  or economic tradability.

## Economic interpretation

- The execution model is a latency/cost sensitivity analysis, not an exact fill
  simulator. It cannot model queue priority, adverse selection or market impact.
- The additive cost grid is transparent but simplified. Actual fees, spread and
  slippage vary with venue tier, order type, volatility and size.
- Extremely high turnover makes the result fragile to sub-basis-point friction; the
  registered strategy's break-even round-trip cost was approximately 0.33 bps.
- The economic result is conditional on the frozen threshold, five-second hold,
  sizing and execution assumptions. It does not prove that every strategy using these
  data is unprofitable.
- Annualised risk ratios and drawdown estimates are unstable over short evaluation
  windows; gross/net totals and break-even cost are more transparent here.

## Engineering interpretation

- The replay speed-up was measured on deterministic synthetic in-memory arrays on one
  Apple-arm64 machine. Compiler, CPU, thermal state and event distribution affect it.
- The benchmark compares the C++ kernel with its deliberately literal pure-Python
  reference, not Polars or the end-to-end research pipeline.
- Benchmark timing excludes disk I/O, feature rolling windows, model fitting and
  economic analysis.
