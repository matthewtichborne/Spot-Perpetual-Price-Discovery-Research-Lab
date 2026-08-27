# Feature dictionary

Status: Phase 3 feature schema version 1.

All predictor columns are calculated on right-labelled 100 ms bars and shifted by
one complete base bar before sampling one-second decision rows. Therefore the
`feature_cutoff_ns` is 100 ms earlier than `decision_time_ns`. Labels are calculated
separately and are never included in the predictor list.

## Decision metadata

| Field | Definition |
|---|---|
| `decision_time_ns` | UTC one-second decision-grid timestamp in nanoseconds |
| `feature_cutoff_ns` | Latest base-bar boundary available to predictors; strictly before the decision |
| `symbol` | Instrument symbol |
| `date` | UTC source partition date |

## Same-market predictors

Each template is emitted for market prefix `spot` and `perpetual`, and for trailing
window suffix `100ms`, `500ms`, `1000ms`, `5000ms`, and `10000ms`.

| Template after `{market}_` | Definition over the trailing window |
|---|---|
| `signed_quantity_{window}` | Sum of aggressor-signed base quantity |
| `signed_notional_{window}` | Sum of aggressor-signed quote notional |
| `quantity_{window}` | Total base quantity |
| `notional_{window}` | Total quote notional |
| `quantity_imbalance_{window}` | Signed quantity divided by quantity; zero without trades, bounded to [-1, 1] |
| `notional_imbalance_{window}` | Signed notional divided by notional; zero without trades, bounded to [-1, 1] |
| `buyer_trades_{window}` | Count of buyer-initiated aggregate trades |
| `seller_trades_{window}` | Count of seller-initiated aggregate trades |
| `trade_count_{window}` | Total aggregate-trade count |
| `arrival_intensity_{window}` | Trade count divided by window length in seconds |
| `average_trade_size_{window}` | Quantity divided by trade count; zero without trades |
| `log_return_{window}` | Log last-price change across the window |
| `realised_volatility_{window}` | Square root of summed squared 100 ms log returns |
| `flow_volatility_interaction_{window}` | Quantity imbalance multiplied by realised volatility |

## Cross-market predictors

| Field/template | Definition |
|---|---|
| `spot_perp_log_basis` | `log(perpetual_last_price / spot_last_price)` |
| `spot_perp_basis_change_{window}` | Change in log basis over each configured window |
| `perpetual_spot_relative_quantity_{window}` | Perpetual quantity divided by spot quantity; null when spot quantity is zero |
| `perpetual_spot_relative_intensity_{window}` | Perpetual trade count divided by spot trade count; null when spot count is zero |
| `spot_perp_basis_zscore_10000ms` | Basis minus its trailing 10 s mean, divided by trailing 10 s standard deviation |

The feature schema contains 157 predictors: 140 same-market predictors and 17
cross-market predictors.

## Labels

For horizons `1000ms`, `5000ms`, and `10000ms`:

| Template | Definition |
|---|---|
| `target_spot_log_return_{horizon}` | Future BTC spot log return from decision time to the horizon |
| `target_spot_direction_{horizon}` | 1 when the future return is positive, otherwise 0; null when the horizon is unavailable |

The final rows of each daily partition have null labels when their future horizon
would cross the partition boundary. No cross-day future price is silently imported.
