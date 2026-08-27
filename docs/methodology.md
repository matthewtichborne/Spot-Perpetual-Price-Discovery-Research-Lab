# Methodology

## Acquisition and normalisation

The acquisition pipeline constructs official Binance daily aggregate-trade URLs from a
strict configuration. Each ZIP and its `.CHECKSUM` sidecar are downloaded without
credentials. Writes are atomic, cached files are reused only after SHA-256
verification, and a corrupted cache entry is redownloaded.

Spot and USD-M futures archives are parsed using separate explicit source schemas.
The CSV member is read directly from the ZIP and is never retained on disk. Source
timestamp scale is validated from the observed integer values: sampled 2025 spot
files use microseconds and sampled futures files use milliseconds. Values are
converted to integer nanoseconds only after the complete file has one consistent
unit and falls inside its configured UTC date.

`is_buyer_maker=true` identifies a seller-initiated trade and maps to aggressor sign
-1; false maps to +1. The pipeline derives notional, signed quantity and signed
notional, sorts by event time and aggregate-trade ID, and atomically writes Zstandard
Parquet under Hive-style symbol, market and date partitions.

DuckDB validates the canonical schema, physical time ordering, positive finite
prices/quantities/notionals, unique aggregate-trade IDs, required values, market
identity, maker/sign correspondence and signed-flow arithmetic. A deterministic JSON
manifest records raw and processed SHA-256 hashes, row counts, source timestamp units
and schema version. This file contains no predictive or trading claims yet.

## Fixed-grid features and labels

Each UTC day is aggregated to right-labelled 100 ms bars. An event is placed at the
first boundary strictly after its timestamp, empty activity fields are zero-filled,
and last trade price is carried forward. Trailing features are calculated at 100,
500, 1,000, 5,000 and 10,000 ms. All predictors are then shifted by one base bar and
sampled at one-second decision times, so `feature_cutoff_ns < decision_time_ns` by
construction.

Same-market features cover signed quantity/notional, normalised flow imbalance,
buyer/seller counts, quantity/notional, arrival intensity, average size, lagged log
returns, realised volatility and flow-volatility interactions. Cross-market features
use the exact shared grid for log basis, basis change/z-score and relative activity.
No forward as-of match is permitted.

Future one-, five- and ten-second BTC spot returns and directions are constructed in
a separate label step. Unavailable daily-tail horizons remain null. The generated
feature validator checks unique chronological decisions, strict feature cutoffs,
finite numeric values and imbalance bounds before atomically writing Parquet.

## Walk-forward analysis

The research design and development, confirmation and final configurations were
SHA-256 recorded before the remaining development archives were downloaded. BTC is
the primary asset; the target is the next five-second spot log return, with
direction as a secondary target. The 30 development days form four expanding folds:
10, 15, 20 and 25 training days, each followed by five evaluation days. A ten-second
purge is removed from the end of training and a ten-second embargo from the start of
evaluation.

The spot-only baseline contains 12 lagged price, signed-flow, activity and realised
volatility variables. The expanded set adds 15 fixed perpetual and cross-market
variables. Median imputation and standardisation are learned independently inside
each training fold. Models comprise training-mean and zero references, unregularised
linear regression, fixed Ridge (`alpha=1`) and fixed logistic regression (`C=1`); no
evaluation fold is used for tuning.

Regression reports include training-mean-referenced OOS R-squared, MSE, MAE and
linear/rank information coefficients. Classification reports include ROC AUC,
precision-recall AUC, Brier score, accuracy and class balance. All are retained by
fold, and primary loss comparisons are also retained by UTC evaluation day. A paired
2,000-replicate day-block bootstrap estimates uncertainty in baseline-minus-expanded
daily MSE. Separate interpretive OLS uses a prespecified parsimonious subset, every
fifth decision row and Newey-West standard errors with 12 lags. This inference is not
used to select or tune the predictive models.

## Confirmation and robustness

Before accessing February confirmation archives, the frozen protocol fixed a
four-candidate XGBoost grid, January-only tuning split, robustness definitions and a
conservative final-selection rule. XGBoost uses histogram trees, expanded features,
every fifth development row and four combinations of depth two/three with 100/200
trees. Its learning rate, sampling fractions, regularisation, seed and all other
settings are fixed. Lowest January 27–31 validation MSE chooses the candidate; it is
then refit on the registered January subsample and scores every February confirmation
row.

Ridge baseline and expanded models are fit on all eligible January observations and
scored on February without refitting. The confirmation suite additionally reports a
within-day 900-row circular-shift negative control for expanded-only features,
low/high volatility regimes using the January median threshold, separate
one/five/ten-second Ridge targets, and an identical ETH Ridge replication. Daily
paired block bootstraps contain 2,000 deterministic replicates.

XGBoost may replace Ridge only when its confirmation OOS R-squared exceeds expanded
Ridge by at least 0.001 and the lower endpoint of the daily Ridge-minus-XGBoost MSE
bootstrap interval is positive. Both conditions held. The resulting XGBoost depth-2,
200-tree expanded specification was frozen for the later one-time holdout evaluation.

## Execution and portfolio analysis

The execution analysis keeps the predictive models fixed and converts their
untouched-confirmation predictions into sparse directional signals. BTC uses the
selected XGBoost model;
ETH uses the prespecified expanded Ridge replication. Each threshold is one standard
deviation of January 27–31 validation predictions, so no February return or execution
result calibrates trading frequency.

For each one-second decision, the engine waits the registered latency and selects the
first spot aggregate trade at or after that timestamp. It exits on the first trade at
least five seconds after entry. A signal is skipped while the previous same-asset
position remains open. Missing eligible entry/exit events are counted, and execution
never falls back to a same-event or cross-day price.

Gross simple return is direction times exit-price divided by entry-price minus one.
Net P&L subtracts configurable additive entry/exit fees, per-side slippage and a
spread proxy. The primary all-in round trip is 7 bps, and 0/2/5/7/10/20 bps are all
reported at 100/500/1,000/2,000/5,000 ms latency. This is a sensitivity surface, not
an order-book fill simulation.

BTC and ETH receive normalised inverse-January-volatility portfolio weights. The
portfolio stops accepting new trades after realised closed-trade net loss reaches 2%
within a UTC day. Fixed sizing is primary; capped inverse-volatility sizing is
secondary. Daily returns, daily Sharpe/Sortino, drawdown, win/loss statistics,
turnover, exposure, break-even cost, regime splits and P&L concentration are computed
from reconciled trade ledgers. Trade-level observations are never annualised.

## Bounded C++ replay

Profiling a 200,000-event pure-Python two-market replay isolated its event loop as the
bounded acceleration target. The reference merges non-decreasing spot/perpetual
streams and emits right-labelled 100 ms base aggregates: carried last price, unsigned
and signed quantity/notional, and buyer/seller/total counts. Events on a grid boundary
enter the next bar, matching the feature pipeline's causal cutoff convention.

The C++20 implementation uses contiguous NumPy views through pybind11, performs the
same merge and aggregation, then returns owned NumPy arrays. It does not accelerate
Parquet reads, feature rolling windows, model fitting or report generation. Parity
tests require identical grids/counts and `rtol=atol=1e-12` for floats, with equal
NaNs. Empty streams, duplicate/tied events, multiple events per bucket, boundary
events, large arrays and invalid ordering are covered.

The frozen benchmark pre-generates one million events per market over one hour and a
36,000-row 100 ms grid. After one warm-up, three calls per implementation are timed
with the same arrays; medians determine throughput and speed-up. Synthetic generation,
imports, parity checking and JSON output are excluded. The result therefore describes
only equivalent in-memory replay-kernel work.

## Final holdout evaluation

Before final access, the evaluation protocol fixed and hashed the sole model, target,
feature list, training sample, predictive metrics and economic assumptions. The
evaluator was tested end to end on synthetic data and refuses to run whenever a final-evaluation
manifest exists. The immutable sealed configuration remains alongside a distinct,
content-hashed one-time open configuration.

XGBoost was refit once on the registered every-five-second BTC rows from development
plus confirmation, then scored every eligible final row. OOS R-squared uses the mean
of all eligible pre-final targets as its reference. Daily metrics and equal-count
prediction deciles were generated without selection.

The economic test inherited the confirmation-calibrated BTC threshold, fixed unit
size, 500 ms latency, five-second hold and 7 bps round-trip cost. Positions do not
overlap; entry and exit use the first observed future spot trades. No final observation was used to choose a
parameter. Gross and net outcomes, their daily paths and exact artifact hashes are
retained in the final manifest.
