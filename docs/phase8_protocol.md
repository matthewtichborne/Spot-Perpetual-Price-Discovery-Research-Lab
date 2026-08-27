# Phase 8 final-holdout protocol

Frozen before final-period access on 2026-08-27.

## Statistical evaluation

- Refit the Phase 5-selected XGBoost/expanded model once on eligible BTCUSDT rows
  from 2025-01-02 through 2025-02-20.
- Use exactly the parameters in `data/manifests/final-model-specification.json` and
  the registered every-five-second training sample.
- Score every eligible BTCUSDT row from 2025-02-21 through 2025-03-02 once for the
  five-second spot log-return target.
- Report training-mean-referenced out-of-sample R-squared, MSE, MAE, Pearson IC,
  rank IC, daily metrics and prediction deciles. Do not tune, calibrate, select an
  alternative model, alter features or inspect another target after opening.

## Economic evaluation

- Use the BTC signal threshold recorded by Phase 6, fixed unit size, first observed
  spot trade at or after 500 ms latency, a five-second holding period and no
  overlapping positions.
- Apply the registered 7 bps all-in round-trip cost. Report gross and net P&L
  separately, including the break-even cost. Final data may not select a cost,
  latency, threshold or sizing rule.

## One-time and reporting rules

`configs/final.yaml` remains sealed as the immutable pre-opening record.
`configs/final-open.yaml` is the sole access configuration. The evaluator refuses
to run if `data/manifests/final-evaluation.json` already exists. All planned metrics
are published regardless of sign, and null or adverse findings remain in the final
report. Raw and processed market data remain untracked.
