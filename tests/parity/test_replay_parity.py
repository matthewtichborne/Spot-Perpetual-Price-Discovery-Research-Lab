from __future__ import annotations

import numpy as np
import pytest

from spot_perp_lab import _replay  # type: ignore[attr-defined]
from spot_perp_lab.replay.cpp import compiler_version, replay_cpp
from spot_perp_lab.replay.reference import (
    MarketEvents,
    empty_market_events,
    replay_reference,
    synthetic_market_events,
)


def _events(times: list[int], prices: list[float], signs: list[float]) -> MarketEvents:
    quantity = np.arange(1, len(times) + 1, dtype=np.float64)
    price = np.array(prices, dtype=np.float64)
    sign = np.array(signs, dtype=np.float64)
    notional = price * quantity
    return MarketEvents(
        event_time_ns=np.array(times, dtype=np.int64),
        aggregate_trade_id=np.arange(len(times), dtype=np.int64),
        price=price,
        quantity=quantity,
        notional=notional,
        signed_quantity=sign * quantity,
        signed_notional=sign * notional,
        is_buyer_maker=sign < 0,
    )


def _assert_parity(
    spot: MarketEvents, perpetual: MarketEvents, start: int, end: int, interval: int
) -> None:
    expected = replay_reference(spot, perpetual, start, end, interval)
    actual = replay_cpp(spot, perpetual, start, end, interval)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].dtype == expected[name].dtype
        if np.issubdtype(expected[name].dtype, np.integer):
            np.testing.assert_array_equal(actual[name], expected[name])
        else:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-12, atol=1e-12, equal_nan=True
            )


def test_cpp_binding_reports_compiler_and_matches_empty_single_and_ties() -> None:
    assert compiler_version()
    empty = empty_market_events()
    _assert_parity(empty, empty, 0, 400, 100)
    _assert_parity(_events([0], [10.0], [1.0]), empty, 0, 400, 100)
    _assert_parity(
        _events([0, 0, 99, 100, 100], [10, 11, 12, 13, 14], [1, -1, 1, -1, 1]),
        _events([0, 50, 100], [20, 21, 22], [-1, 1, -1]),
        0,
        400,
        100,
    )


def test_cpp_matches_reference_on_large_fixed_array_without_mutation() -> None:
    start = 1_735_776_000_000_000_000
    end = start + 60_000_000_000
    spot = synthetic_market_events(50_000, start, end, 17, 100_000.0)
    perpetual = synthetic_market_events(50_000, start, end, 23, 100_010.0)
    spot_times = spot.event_time_ns.copy()
    _assert_parity(spot, perpetual, start, end, 100_000_000)
    np.testing.assert_array_equal(spot.event_time_ns, spot_times)


def test_cpp_rejects_unequal_lengths_out_of_order_and_invalid_grid() -> None:
    integer = np.array([1, 2], dtype=np.int64)
    floating = np.array([1.0, 2.0], dtype=np.float64)
    maker = np.array([False, True], dtype=np.bool_)
    empty_i = np.array([], dtype=np.int64)
    empty_f = np.array([], dtype=np.float64)
    empty_b = np.array([], dtype=np.bool_)

    def call(times: np.ndarray, quantity: np.ndarray, end: int = 10) -> object:
        return _replay.replay_two_markets(
            times,
            integer,
            floating,
            quantity,
            floating,
            floating,
            floating,
            maker,
            empty_i,
            empty_i,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_b,
            0,
            end,
            1,
        )

    with pytest.raises(ValueError, match="equal lengths"):
        call(integer, np.array([1.0]))
    with pytest.raises(ValueError, match="non-decreasing"):
        call(np.array([2, 1], dtype=np.int64), floating)
    with pytest.raises(ValueError, match="whole intervals"):
        call(integer, floating, end=0)
