"""Pure-Python reference for the bounded two-market replay component."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Int64Array = NDArray[np.int64]
Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MarketEvents:
    """Columnar canonical events for one time-ordered market stream."""

    event_time_ns: Int64Array
    aggregate_trade_id: Int64Array
    price: Float64Array
    quantity: Float64Array
    notional: Float64Array
    signed_quantity: Float64Array
    signed_notional: Float64Array
    is_buyer_maker: BoolArray

    def __post_init__(self) -> None:
        lengths = {
            len(self.event_time_ns),
            len(self.aggregate_trade_id),
            len(self.price),
            len(self.quantity),
            len(self.notional),
            len(self.signed_quantity),
            len(self.signed_notional),
            len(self.is_buyer_maker),
        }
        if len(lengths) != 1:
            raise ValueError("market event columns must have equal lengths")
        if len(self.event_time_ns) > 1 and np.any(self.event_time_ns[1:] < self.event_time_ns[:-1]):
            raise ValueError("market event timestamps must be non-decreasing")

    @property
    def size(self) -> int:
        return len(self.event_time_ns)


def empty_market_events() -> MarketEvents:
    """Return a correctly typed empty canonical stream."""

    return MarketEvents(
        event_time_ns=np.array([], dtype=np.int64),
        aggregate_trade_id=np.array([], dtype=np.int64),
        price=np.array([], dtype=np.float64),
        quantity=np.array([], dtype=np.float64),
        notional=np.array([], dtype=np.float64),
        signed_quantity=np.array([], dtype=np.float64),
        signed_notional=np.array([], dtype=np.float64),
        is_buyer_maker=np.array([], dtype=np.bool_),
    )


def _validate_bounds(
    spot: MarketEvents,
    perpetual: MarketEvents,
    start_ns: int,
    end_ns: int,
    interval_ns: int,
) -> int:
    if interval_ns <= 0 or end_ns <= start_ns or (end_ns - start_ns) % interval_ns:
        raise ValueError("grid bounds must define positive whole intervals")
    for events in (spot, perpetual):
        if events.size and (
            int(events.event_time_ns[0]) < start_ns or int(events.event_time_ns[-1]) >= end_ns
        ):
            raise ValueError("market event timestamp falls outside replay grid")
    return (end_ns - start_ns) // interval_ns


def replay_reference(
    spot: MarketEvents,
    perpetual: MarketEvents,
    start_ns: int,
    end_ns: int,
    interval_ns: int,
) -> dict[str, Int64Array | Float64Array]:
    """Merge two streams and emit right-labelled fixed-grid base aggregates.

    Equal cross-market timestamps process spot first. Duplicate timestamps and IDs are
    retained as distinct aggregate-trade events. Each event is assigned to the first
    grid boundary strictly after its timestamp.
    """

    bars = _validate_bounds(spot, perpetual, start_ns, end_ns, interval_ns)
    output: dict[str, Int64Array | Float64Array] = {
        "decision_time_ns": start_ns + np.arange(1, bars + 1, dtype=np.int64) * interval_ns
    }
    for prefix in ("spot", "perpetual"):
        output[f"{prefix}_last_price"] = np.full(bars, np.nan, dtype=np.float64)
        for name in (
            "quantity",
            "notional",
            "signed_quantity",
            "signed_notional",
        ):
            output[f"{prefix}_{name}"] = np.zeros(bars, dtype=np.float64)
        for name in ("trade_count", "buyer_trade_count", "seller_trade_count"):
            output[f"{prefix}_{name}"] = np.zeros(bars, dtype=np.int64)

    indices = [0, 0]
    streams = (spot, perpetual)
    prefixes = ("spot", "perpetual")
    while indices[0] < spot.size or indices[1] < perpetual.size:
        if indices[1] >= perpetual.size:
            market = 0
        elif indices[0] >= spot.size:
            market = 1
        else:
            spot_time = int(spot.event_time_ns[indices[0]])
            perpetual_time = int(perpetual.event_time_ns[indices[1]])
            market = 0 if spot_time <= perpetual_time else 1
        events = streams[market]
        prefix = prefixes[market]
        event = indices[market]
        timestamp = int(events.event_time_ns[event])
        bucket = (timestamp - start_ns) // interval_ns
        output[f"{prefix}_last_price"][bucket] = events.price[event]
        output[f"{prefix}_quantity"][bucket] += events.quantity[event]
        output[f"{prefix}_notional"][bucket] += events.notional[event]
        output[f"{prefix}_signed_quantity"][bucket] += events.signed_quantity[event]
        output[f"{prefix}_signed_notional"][bucket] += events.signed_notional[event]
        output[f"{prefix}_trade_count"][bucket] += 1
        count_name = "seller_trade_count" if events.is_buyer_maker[event] else "buyer_trade_count"
        output[f"{prefix}_{count_name}"][bucket] += 1
        indices[market] += 1

    for prefix in ("spot", "perpetual"):
        prices = output[f"{prefix}_last_price"]
        last = np.nan
        for index in range(bars):
            if np.isnan(prices[index]):
                prices[index] = last
            else:
                last = prices[index]
    return output


def synthetic_market_events(
    events: int,
    start_ns: int,
    end_ns: int,
    seed: int,
    price_origin: float,
) -> MarketEvents:
    """Generate a deterministic sorted benchmark stream."""

    generator = np.random.default_rng(seed)
    timestamps = np.sort(
        generator.integers(start_ns, end_ns, size=events, dtype=np.int64),
        kind="stable",
    )
    prices = price_origin * np.exp(np.cumsum(generator.normal(0.0, 1e-6, size=events)))
    quantities = generator.lognormal(mean=-2.0, sigma=0.8, size=events)
    signs = generator.choice(np.array([-1.0, 1.0]), size=events)
    notionals = prices * quantities
    return MarketEvents(
        event_time_ns=timestamps,
        aggregate_trade_id=np.arange(events, dtype=np.int64),
        price=prices.astype(np.float64),
        quantity=quantities.astype(np.float64),
        notional=notionals.astype(np.float64),
        signed_quantity=(signs * quantities).astype(np.float64),
        signed_notional=(signs * notionals).astype(np.float64),
        is_buyer_maker=(signs < 0),
    )
