# Limitations

- Aggregate trades do not expose the full order book, quoted spread or queue position.
- Signed aggregate-trade flow is not limit-order-book order-flow imbalance.
- The execution model is a latency/cost sensitivity analysis, not an exact fill
  simulator.
- Binance-specific findings may not generalise across venues or market regimes.
- Phase 4 covers only BTC and one 30-day development regime. Its positive incremental
  metrics require confirmation on the prespecified later period and ETH replication.
- Five-second labels overlap at one-second decision frequency. Fold/day reporting,
  a day-block bootstrap and HAC errors address dependence only approximately.
- The development analysis compares several prespecified model/scope combinations;
  its preferred specification is consequently a selection result, not an untouched
  confirmatory estimate.
- HAC coefficients describe conditional association under a linear specification;
  they do not establish price-discovery causality.
- Aggregate-trade archive timing, clock alignment and Binance-specific microstructure
  can differ from executable real-time information.
- OOS R-squared, information coefficient and AUC are statistical metrics. No Phase 4
  result includes fees, bid/ask spread, slippage, latency, market impact, funding,
  inventory constraints or position sizing, so no claim of economic tradability is
  made.
- The final holdout was evaluated exactly once under the frozen Phase 8 protocol.
  Its ten contiguous days provide an untouched estimate, but not broad temporal or
  cross-venue generalisation.
- Phase 5 confirmation is later than development but is still one contiguous 20-day
  Binance regime. ETH replication provides cross-asset evidence, not cross-venue or
  long-horizon evidence.
- The nonlinear model-selection rule was evaluated on confirmation, so its selected
  XGBoost estimate is not a substitute for the still-sealed final evaluation.
- The circular-shift placebo is one negative control; it cannot eliminate every clock,
  feed-construction or common-information explanation.
- Stronger one-second than ten-second statistical metrics may reflect short-lived
  common reactions as well as perpetual-to-spot price discovery.
- Phase 6 uses observed aggregate-trade prices, not quotes or an order book. It cannot
  model queue priority, available depth, adverse selection or market impact, so even
  the zero-cost gross result is not an executable performance claim.
- The all-in additive cost grid is deliberately transparent but simplified. Actual
  fees, spread and slippage vary with venue tier, order type, volatility and size.
- Extremely high turnover makes the result fragile to sub-basis-point frictions; the
  registered portfolio break-even cost is only about 0.34 bps.
- The 2% daily portfolio stop mechanically truncates activity after costs overwhelm
  gross P&L. Risk metrics based on only 20 confirmation days, especially annualised
  Sharpe and Sortino, are unstable and should not be treated as long-run estimates.
- The Phase 6 economic null is conditional on the frozen threshold, five-second hold,
  sizing and execution assumptions. It does not prove that every possible strategy
  using these data is unprofitable.
- The Phase 7 speed-up is measured on deterministic synthetic in-memory arrays on one
  Apple-arm64 machine. Compiler, CPU, thermal state and event distribution affect it.
- The benchmark compares the C++ kernel with its deliberately literal pure-Python
  reference, not with Polars. It must not be interpreted as an 81.76x end-to-end
  pipeline improvement.
- Extension build and parity are tested through Python/pybind11; the benchmark does
  not measure disk I/O, allocation outside the call, feature rolling windows, models
  or economic analysis.
- The final positive OOS R-squared and information coefficients establish forecast
  association, not causality or tradability. The registered economic result is
  strongly negative after costs and should not be reframed as a trading success.
- Final annualised Sharpe/Sortino and drawdown statistics are unstable over ten days;
  the gross/net totals and break-even cost are more transparent for this experiment.
