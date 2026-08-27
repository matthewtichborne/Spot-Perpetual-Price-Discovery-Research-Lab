from __future__ import annotations

import numpy as np
import pytest

from spot_perp_lab.replay.reference import MarketEvents, empty_market_events, replay_reference


def _events(times: list[int], prices: list[float], signs: list[float]) -> MarketEvents:
    quantity = np.ones(len(times), dtype=np.float64)
    price = np.array(prices, dtype=np.float64)
    sign = np.array(signs, dtype=np.float64)
    return MarketEvents(
        event_time_ns=np.array(times, dtype=np.int64),
        aggregate_trade_id=np.arange(len(times), dtype=np.int64),
        price=price,
        quantity=quantity,
        notional=price,
        signed_quantity=sign,
        signed_notional=sign * price,
        is_buyer_maker=sign < 0,
    )


def test_empty_and_single_event_replay() -> None:
    empty = replay_reference(empty_market_events(), empty_market_events(), 0, 300, 100)
    assert empty["decision_time_ns"].tolist() == [100, 200, 300]
    assert np.isnan(empty["spot_last_price"]).all()
    single = replay_reference(_events([0], [10.0], [1.0]), empty_market_events(), 0, 300, 100)
    assert single["spot_last_price"].tolist() == [10.0, 10.0, 10.0]
    assert single["spot_trade_count"].tolist() == [1.0, 0.0, 0.0]
    assert single["spot_trade_count"].dtype == np.int64


def test_ties_duplicates_and_multiple_events_in_bucket_are_retained() -> None:
    spot = _events([0, 0, 99, 100], [10.0, 11.0, 12.0, 13.0], [1.0, -1.0, 1.0, -1.0])
    perpetual = _events([0, 100], [20.0, 21.0], [-1.0, 1.0])
    result = replay_reference(spot, perpetual, 0, 300, 100)
    assert result["spot_trade_count"].tolist() == [3.0, 1.0, 0.0]
    assert result["spot_last_price"].tolist() == [12.0, 13.0, 13.0]
    assert result["spot_buyer_trade_count"].tolist() == [2.0, 0.0, 0.0]
    assert result["spot_seller_trade_count"].tolist() == [1.0, 1.0, 0.0]
    assert result["perpetual_trade_count"].tolist() == [1.0, 1.0, 0.0]


def test_out_of_order_input_and_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        _events([2, 1], [10.0, 11.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="whole intervals"):
        replay_reference(empty_market_events(), empty_market_events(), 0, 250, 100)
    with pytest.raises(ValueError, match="outside"):
        replay_reference(_events([300], [10.0], [1.0]), empty_market_events(), 0, 300, 100)
