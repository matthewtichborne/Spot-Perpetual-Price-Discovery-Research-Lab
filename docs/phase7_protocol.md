# Frozen Phase 7 replay and benchmark protocol

Status: **frozen on 2026-08-27 after profiling the Python reference and before
compiling the C++ implementation**. The final holdout remains sealed.

## Bounded component

The component merges two already-normalised, non-decreasing spot and perpetual event
streams and emits right-labelled fixed-grid base aggregates. It does not replace the
full research pipeline or claim to accelerate modelling, I/O or Polars.

Inputs for each market are int64 event time and aggregate-trade ID, float64 price,
quantity, notional, signed quantity and signed notional, and boolean buyer-maker
flags. The output is a shared int64 decision-time grid plus, for each market, last
price, quantity/notional sums, signed sums, total count and buyer/seller counts.

Every event is retained, including duplicate timestamps and IDs. Events enter the
first grid boundary strictly after their timestamp. Same-market input order decides
last price for tied timestamps; equal cross-market timestamps process spot first.
Empty bars have zero activity, and last price carries forward while remaining NaN
before the first event. Decreasing timestamps, unequal column lengths, out-of-range
events and invalid grids are rejected.

## Parity requirements

The pure-Python reference is the specification. Python and C++ must return identical
column names, shapes and integer grids/counts. Float arrays use
`rtol=1e-12`, `atol=1e-12`, with NaNs equal. Tests cover empty and single streams,
duplicates, ties, multiple events per bucket, boundary timestamps, unequal lengths
and out-of-order inputs. A larger fixed array must pass parity before timing.

## Fixed benchmark

- Workload: 1,000,000 spot plus 1,000,000 perpetual events.
- UTC-like nanosecond window: one hour beginning at
  `1735776000000000000`.
- Grid: 100,000,000 ns (100 ms), producing 36,000 rows.
- Synthetic seeds: 17 spot and 23 perpetual; price origins 100,000 and 100,010.
- Python and C++ receive the same pre-generated contiguous NumPy arrays.
- Build: C++20 Release configuration through CMake/pybind11.
- Timing: one untimed warm-up followed by three timed repetitions per
  implementation, in the same process; report the median.
- Runtime covers only the equivalent merge/aggregation call. Synthetic generation,
  parity checking, imports and JSON/report writing are outside the timed region.
- Throughput is total input events divided by median seconds. Speed-up is Python
  median divided by C++ median.

The report records OS, architecture, processor, Python, compiler, build type, event
count, grid rows, per-repeat durations, medians, throughput and speed-up. Results
describe only this bounded equivalent-work kernel and may not be compared with
Polars or end-to-end runtime without a separate equivalent benchmark.

## Pre-C++ profile

The fixed 100,000-event-per-market profile completed before C++ compilation. Its
content hash is
`358f9b4fa99bec223dcaeabfbc122789dea44877062f56a3ac285046a7e68521`;
the Python event loop consumed essentially all recorded cumulative runtime.
