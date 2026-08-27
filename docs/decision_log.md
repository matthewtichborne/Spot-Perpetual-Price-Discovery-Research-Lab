# Decision log

## 2026-08-25 — Proceed after Phase 0 feasibility spike

- **Decision:** Proceed with Binance public daily aggregate-trade archives and begin
  the repository scaffold.
- **Evidence:** Credential-free downloads succeeded, official SHA-256 checksums
  matched, and BTC/ETH spot/perpetual archives overlapped on two consecutive dates.
- **Consequence:** Parsing must be market-specific. Timestamp units are validated
  before conversion to nanoseconds because sampled 2025 spot data is in microseconds
  and sampled USD-M futures data is in milliseconds.

## 2026-08-25 — Keep the final holdout sealed

- **Decision:** `configs/final.yaml` is a schema-valid placeholder marked `sealed`;
  it does not authorise holdout download or evaluation.
- **Reason:** The hypothesis, feature set, splits and model-selection rule have not
  yet been frozen.

## 2026-08-25 — Use a small typed Phase 1 core

- **Decision:** Start with strict Pydantic/YAML configuration, Typer CLI, pure archive
  URL construction and timestamp-scale validation.
- **Reason:** These components make the feasibility evidence executable and testable
  without prematurely adding the modelling stack or downloading data in CI.

## 2026-08-25 — Freeze canonical trade schema version 1

- **Decision:** Normalise verified archives to Zstandard Parquet partitioned by
  `symbol`, `market_type` and UTC `date`, with all event timestamps stored as int64
  nanoseconds and signed aggregate-trade flow derived from `is_buyer_maker`.
- **Evidence:** The live two-day BTC smoke run processed and DuckDB-validated
  4,864,162 events across four partitions. A repeat run used all four cached archives
  and reproduced manifest hash
  `394e8274aa0a2ce38d1d9fba66d3b1a7d012b6dda91291d7ab7937a5bf8701bf`.
- **Consequence:** Phase 3 feature code can depend on the versioned column names and
  nanosecond event-time contract. Schema changes require a new version.

## 2026-08-25 — Keep large data out of Git and small manifests in Git

- **Decision:** Raw ZIPs and processed Parquet remain ignored; deterministic JSON
  manifests are committed.
- **Reason:** The smoke data occupies approximately 66 MiB raw and 106 MiB processed,
  while the manifest is small and preserves reproducible provenance.

## 2026-08-25 — Adopt a conservative 100 ms feature cutoff

- **Decision:** Use 100 ms right-labelled base bars, shift every predictor by one
  complete base bar, and sample one-second decisions. Keep labels unlagged and
  structurally separate from the 157-column predictor list.
- **Reason:** This makes the latest allowed information explicit in
  `feature_cutoff_ns` and removes boundary ambiguity while preserving the planned
  subsecond flow windows.
- **Evidence:** Hand-worked boundary, rolling-flow, basis, horizon and future-event
  tests pass. The live two-day build produced 172,793 validated decision rows and six
  labels without importing cross-day future prices. A complete repeat build produced
  byte-identical Parquet files and feature manifest hash
  `c2b991aec9bf8b6f4b1da1a8ea801eeb849f0361a56a75eb053cb4d814da00c4`.

## 2026-08-25 — Bound numerical features by mathematical identities

- **Decision:** Clip realised variance at zero before its square root and flow
  imbalance to [-1, 1] when trades exist; define empty-window imbalance and average
  size as zero.
- **Reason:** Visual and automated QA detected floating-point rolling-sum residue in
  otherwise empty windows. The bounds encode identities, not winsorisation of market
  observations.

## 2026-08-25 — Freeze Phase 4 before completing development acquisition

- **Decision:** Freeze the primary hypothesis, target, 12-feature spot baseline,
  27-feature expanded specification, four walk-forward folds, fixed model parameters,
  uncertainty procedure and selection rule before downloading the remaining
  development dates.
- **Evidence:** `data/manifests/research-design-freeze.json` records SHA-256 hashes for
  the research design and development, confirmation and final configurations. The
  final configuration remains marked `sealed`.
- **Consequence:** Phase 4 results may select only among the registered linear
  specifications. Phase 5 confirmation and the final holdout cannot be redefined in
  response to these development results.

## 2026-08-26 — Complete Phase 4 research baselines

- **Decision:** Carry `ridge` / `expanded` forward as the preferred frozen linear
  specification under the registered mean-fold-OOS-R-squared rule.
- **Evidence:** Across four purged expanding folds and 20 evaluation days, mean fold
  OOS R-squared was 0.016964 versus 0.007310 for Ridge/baseline. Expanded daily MSE
  was lower on 19/20 days, and the paired 2,000-replicate day-block bootstrap interval
  for mean daily improvement was [1.697e-10, 5.864e-10]. Mean fixed-logistic ROC AUC
  was 0.626996 versus 0.585761; no model failures were recorded.
- **Provenance:** The Phase 4 manifest hash is
  `3043e85c631ce88bc320ee340f8493de38cba71b1e8b730815e5e00f3afe7dc0`.
- **Consequence:** These development-only results motivate Phase 5 confirmation but
  do not establish causality or tradability. Fees, spread, latency, impact and risk
  rules remain deferred to Phase 6; the final holdout remains unopened.

## 2026-08-26 — Freeze Phase 5 before confirmation access

- **Decision:** Register the January-only four-candidate XGBoost search, 900-second
  alignment placebo, fixed volatility regimes, one/five/ten-second horizons, ETH
  replication and final-selection rule before downloading February confirmation
  archives.
- **Evidence:** `data/manifests/phase5-protocol-freeze.json` records that no
  confirmation or final-period file was present and hashes the protocol and all
  relevant configurations.
- **Consequence:** Confirmation cannot tune XGBoost or redefine the robustness checks;
  the final-period configuration remains sealed.

## 2026-08-26 — Select the final XGBoost specification after Phase 5

- **Decision:** Freeze XGBoost/expanded with `max_depth=2`, `n_estimators=200` and the
  registered fixed parameters for the later one-time final evaluation.
- **Evidence:** BTC confirmation OOS R-squared was 0.030302 for XGBoost/expanded,
  0.024278 for Ridge/expanded and 0.010312 for Ridge/baseline. The paired daily
  Ridge-minus-XGBoost MSE interval was [1.604e-10, 2.446e-10], satisfying both frozen
  selection conditions. The alignment placebo was baseline-like (0.010231), and ETH
  Ridge OOS R-squared was 0.013073 expanded versus 0.004566 baseline.
- **Provenance:** Phase 5 manifest hash
  `cb6ba4b23fd5e647a2e29ab5eee1169c5c53049b6b80b40911f2ac435390b97f`;
  final specification hash
  `fe69b128d018481b8db3e663668e012f3d5bd964f68316d9d0d1fde36deac570`.
- **Consequence:** Robustness results cannot alter the selected specification. The
  final holdout remains sealed, and no tradability claim is made before Phase 6 costs
  and execution analysis.

## 2026-08-26 — Freeze Phase 6 before economic analysis

- **Decision:** Register the January-calibrated signal threshold, first-future-trade
  fill rule, five-second hold, five latencies, six all-in costs, non-overlap rule,
  sizing experiments, portfolio risk weights and daily loss limit before running an
  execution backtest.
- **Evidence:** `data/manifests/phase6-protocol-freeze.json` hashes the protocol,
  configuration, Phase 5 result and final model specification and records that no
  Phase 6 economic result existed.
- **Consequence:** Costs, latency and thresholds cannot be selected to improve the
  observed confirmation result.

## 2026-08-26 — Retain the Phase 6 economic null

- **Decision:** Report that the frozen statistical signal does not survive the
  registered execution-cost assumption; do not redesign the signal after seeing the
  cost surface.
- **Evidence:** At 500 ms and 7 bps, the risk-limited combined portfolio accepted
  1,271 trades and generated +0.020998 gross P&L but −0.407156 net P&L. Its
  break-even round-trip cost was 0.343302 bps. At zero cost, gross P&L declined from
  8.165207 at 100 ms to 1.042226 at 5 seconds. Fixed-size BTC and ETH legs were net
  negative at every positive registered cost, and inverse-volatility sizing did not
  change that conclusion.
- **Provenance:** Phase 6 manifest hash
  `4fb42df4d371786a489d2eeb3f1c502743715b5c848c2c2d3b9924d16cd8acff`.
- **Consequence:** Statistical predictability is not presented as tradable alpha.
  Phase 7 may improve replay performance but cannot change this economic conclusion;
  the final holdout remains sealed for Phase 8.

## 2026-08-27 — Bound Phase 7 to equivalent replay work

- **Decision:** Accelerate only the profiled two-market merge and fixed-grid base
  aggregation, retaining a pure-Python specification and excluding I/O, rolling
  features, modelling and reporting from the claim.
- **Evidence:** The pre-C++ 200,000-event profile attributed essentially all recorded
  cumulative time to the Python replay loop. `data/manifests/phase7-protocol-freeze.json`
  hashes the reference, profile and benchmark method before compilation.
- **Consequence:** Any speed-up must be described as a bounded-kernel result and may
  not be compared with Polars or presented as end-to-end acceleration.

## 2026-08-27 — Accept the C++20 replay kernel

- **Decision:** Retain the C++20/pybind11 implementation after Python/C++ parity and
  the frozen equivalent-work benchmark passed.
- **Evidence:** On two million events and 36,000 output rows, median Python runtime was
  3.568878 s (560,400 events/s) and median C++ runtime was 0.043651 s (45.8 million
  events/s), an 81.76x bounded-kernel speed-up. Integer outputs were exact and floats
  agreed within `1e-12`. Empty, single, duplicate, tied, multi-bucket, out-of-order and
  large-array cases passed.
- **Provenance:** Phase 7 manifest hash
  `d1090ab691d787338dee94a11f9b1837b615747572c6dfe486fc17d8e95a89b5`.
- **Consequence:** CI now compiles and imports the extension before parity tests. This
  engineering result does not alter the Phase 6 economic null or open the final
  holdout.

## 2026-08-27 — Open the final holdout exactly once

- **Decision:** Authorise the sole Phase 8 evaluation only after freezing and hashing
  the protocol, open configuration, evaluator, final model specification and inherited
  Phase 6 execution assumptions.
- **Evidence:** Before access, no final artifact existed and the synthetic evaluator
  test plus the complete 54-test suite passed. Forty final archives containing
  88,965,078 trades then passed checksum and structural validation, and 20 feature
  partitions reconciled to their source manifest.
- **Consequence:** The final-evaluation manifest activates the one-time guard. No
  alternative model, horizon, feature, threshold, latency or cost may be selected
  from the observed final results.

## 2026-08-27 — Retain both final conclusions

- **Decision:** Report the positive predictive result and negative economic result
  together, without tuning after opening.
- **Evidence:** Across 863,926 final observations, XGBoost/expanded achieved OOS
  R-squared 0.017629, Pearson IC 0.133406 and rank IC 0.198230. The registered rule
  produced +2.848618 gross P&L across 86,405 trades, but −57.634882 after the frozen
  7 bps round-trip cost; break-even cost was 0.329682 bps.
- **Provenance:** Final evaluation-manifest hash
  `3c2edcd0787106eb990af0d44ca7ffd8922cdead33c87d3cf2f669931a12d0f7`.
- **Consequence:** The project supports short-horizon forecast association but makes
  no claim of causality or economic tradability under realistic frictions.
