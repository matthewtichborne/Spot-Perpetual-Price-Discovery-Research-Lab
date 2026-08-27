# Data dictionary

## Canonical normalised trade event (schema version 1)

| Field | Type | Definition |
|---|---|---|
| `exchange` | string | `binance` |
| `market_type` | categorical | `spot` or `perpetual` |
| `symbol` | categorical | `BTCUSDT` or `ETHUSDT` |
| `event_time_ns` | int64 | Validated source timestamp converted to UTC nanoseconds |
| `aggregate_trade_id` | int64 | Source aggregate-trade identifier |
| `first_trade_id` | int64 | First underlying source trade identifier |
| `last_trade_id` | int64 | Last underlying source trade identifier |
| `price` | float64 | Trade price |
| `quantity` | float64 | Base-asset quantity |
| `notional` | float64 | `price * quantity` |
| `is_buyer_maker` | bool | Source flag; true means seller-initiated trade |
| `aggressor_sign` | int8 | +1 buyer-initiated, -1 seller-initiated |
| `signed_quantity` | float64 | `aggressor_sign * quantity` |
| `signed_notional` | float64 | `aggressor_sign * notional` |
| `source_file` | string | Original archive filename |
