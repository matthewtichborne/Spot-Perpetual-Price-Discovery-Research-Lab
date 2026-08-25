# Phase 0 feasibility report

Date checked: 2026-08-25 (Europe/London)
Decision: **GO** for Phase 1 and a two-day BTC ingestion smoke test.

## Access and provenance

Binance's official public archive served daily aggregate-trade ZIP files over HTTPS
without credentials. The archive documentation states that daily data is published the
following day and supplies a `.CHECKSUM` SHA-256 sidecar for each file.

Evidence was collected from the official
[`binance-public-data` documentation](https://github.com/binance/binance-public-data/blob/master/README.md)
and these two downloaded files:

| Market | Archive date | ZIP bytes | CSV bytes | Rows including header where present | SHA-256 verified |
|---|---:|---:|---:|---:|---|
| BTCUSDT spot | 2025-01-02 | 19,186,271 | 112,383,251 | 1,299,165 | yes |
| BTCUSDT USD-M perpetual | 2025-01-02 | 18,947,850 | 99,961,710 | 1,505,249 | yes |

The observed hashes were:

- Spot: `f7f032f4eef277809ac0d688eb27b48767123211cd938185c171906c4208118c`
- Perpetual: `7714f1084931a01b1474db471e9cebd747575dbccc6c94dbb1b289b317128f55`

Both exactly matched their official `.CHECKSUM` files. Sample archives were kept only
in a temporary feasibility directory and are not committed.

## Raw schemas and timestamp evidence

The sampled spot CSV has **no header** and eight fields:

```text
aggregate_trade_id, price, quantity, first_trade_id, last_trade_id,
transact_time, is_buyer_maker, is_best_match
```

Its first timestamp was `1735776000113701` (16 digits, microseconds). This agrees
with Binance's documented switch of spot archive timestamps to microseconds from
2025-01-01 onward.

The sampled USD-M futures CSV has a header and seven fields:

```text
agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
```

Its first timestamp was `1735776005115` (13 digits, milliseconds). Normalisation
must therefore validate each source unit and convert both to integer nanoseconds;
it must not apply one hard-coded multiplier to every file.

## Market/date overlap

HTTP metadata requests returned `200 OK` for all four combinations—BTCUSDT and
ETHUSDT, spot and USD-M perpetual—on both 2025-01-02 and 2025-01-03. The two-day
BTC smoke period and a later BTC/ETH development period are therefore available.

| Date | BTC spot | BTC perpetual | ETH spot | ETH perpetual | Total ZIP bytes |
|---|---:|---:|---:|---:|---:|
| 2025-01-02 | 19,186,271 | 18,947,850 | 12,389,828 | 18,857,707 | 69,381,656 |
| 2025-01-03 | 15,140,932 | 13,286,853 | 11,168,224 | 16,658,252 | 56,254,261 |

## Storage estimate

The two-day mean across all four series was 62,817,958.5 compressed bytes/day.
The sampled BTC CSV-to-ZIP ratio was 5.57×. Activity varies substantially, so these
are planning estimates rather than capacity guarantees.

| Period | Estimated ZIP storage | Approximate extracted CSV storage |
|---:|---:|---:|
| 30 days | 1.76 GiB | 9.77 GiB |
| 60 days | 3.51 GiB | 19.55 GiB |
| 90 days | 5.27 GiB | 29.32 GiB |

Verified raw CSVs should be converted to Parquet and removed after successful
normalisation, avoiding persistent extracted-CSV storage.

## Risks and fallback

- Historical spot files straddle a timestamp-unit change at 2025-01-01.
- Spot and futures archives differ in header presence and field count.
- Daily activity and file size are non-stationary; storage needs a safety margin.
- Aggregate trades support signed trade-flow research, not limit-order-book claims or
  exact fill simulation.

If the official archive later becomes unavailable, stop before changing sources.
The compliant fallback is Binance's public market-data REST endpoints for a bounded
period, with the provenance and acquisition method recorded; it is not a silent
substitution and would require a new feasibility decision.
