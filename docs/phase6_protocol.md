# Frozen Phase 6 execution protocol

Status: **frozen on 2026-08-26 before running any economic backtest**. The final
holdout remains sealed. Phase 6 uses only the already-open January development and
February confirmation samples.

## Signals and evaluation sample

- BTC uses the frozen Phase 5 XGBoost/expanded five-second model: depth 2, 200 trees
  and the fixed parameters in `data/manifests/final-model-specification.json`.
- For an out-of-sample economic test, BTC is trained on January and predicts the
  untouched February 1–20 confirmation sample exactly as in Phase 5.
- ETH uses the prespecified Ridge/expanded replication model trained on January and
  evaluated on February. It is a portfolio robustness leg, not the selected final
  BTC model.
- Each asset's symmetric trading threshold is one standard deviation of predictions
  from the fixed January 27–31 validation block. A prediction above the positive
  threshold opens long; below the negative threshold opens short; otherwise it is
  flat. February never calibrates a threshold.
- The reference position is constant unit gross exposure. A secondary inverse-
  volatility experiment multiplies exposure by the January median five-second spot
  realised volatility divided by current lagged volatility, clipped to [0.25, 2.0].
  The resulting position is then capped at the configured maximum gross exposure of
  1.0, so this secondary rule can reduce but never lever the reference position.

## Timing and fills

- Signals occur at one-second decision timestamps and retain the Phase 3 causal
  feature cutoff.
- Latencies are fixed to 100, 500, 1,000, 2,000 and 5,000 milliseconds; 500 ms is the
  primary scenario.
- Entry is the first observed spot aggregate trade at or after decision time plus
  latency. Entry can never use the feature/decision event.
- Exit is the first observed spot aggregate trade at or after entry time plus five
  seconds.
- Positions are short, flat or long. Positions may not overlap within an asset; a
  new signal is considered only after the prior observed exit timestamp.
- A missing eligible entry or exit is retained as a skipped-fill count, never filled
  from another day.

## Costs and risk

The reference all-in round-trip cost is 7 bps: 2 bps entry fee, 2 bps exit fee, 1
bps slippage per side and a conservative 0.5 bps spread proxy per side. This is an
additive sensitivity model, not an order-book fill simulator. The registered all-in
round-trip sensitivity grid is 0, 2, 5, 7, 10 and 20 bps.

BTC and ETH portfolio risk weights are inverse January median five-second realised
volatility and normalised to total gross exposure 1.0. The combined portfolio accepts
no new trades after cumulative marked closed-trade net loss reaches 2% within a UTC
day. Per-asset positions remain capped by their normalised risk weight.

## Reports and interpretation

For BTC, ETH and their combined portfolio, reports retain trade count, holding time,
exposure, turnover, gross/net P&L, daily returns, annualised daily Sharpe, Sortino,
maximum drawdown, win rate, average win/loss, break-even round-trip cost, P&L
concentration, volatility regime results and degradation across the full latency and
cost grids. Trade-level returns are never annualised.

The primary economic conclusion uses 500 ms latency, fixed sizing and 7 bps cost.
Gross and net results are always separate. Daily P&L must reconcile exactly to the
trade ledger. No Phase 6 result changes the frozen predictive model or opens the final
holdout.
