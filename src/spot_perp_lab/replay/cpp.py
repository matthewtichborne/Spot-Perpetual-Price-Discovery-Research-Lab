"""Typed Python wrapper for the optional compiled replay kernel."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spot_perp_lab import _replay  # type: ignore[attr-defined]
from spot_perp_lab.replay.reference import MarketEvents


def replay_cpp(
    spot: MarketEvents,
    perpetual: MarketEvents,
    start_ns: int,
    end_ns: int,
    interval_ns: int,
) -> dict[str, NDArray[np.int64] | NDArray[np.float64]]:
    """Call the C++20 replay kernel with canonical NumPy columns."""

    result = _replay.replay_two_markets(
        spot.event_time_ns,
        spot.aggregate_trade_id,
        spot.price,
        spot.quantity,
        spot.notional,
        spot.signed_quantity,
        spot.signed_notional,
        spot.is_buyer_maker,
        perpetual.event_time_ns,
        perpetual.aggregate_trade_id,
        perpetual.price,
        perpetual.quantity,
        perpetual.notional,
        perpetual.signed_quantity,
        perpetual.signed_notional,
        perpetual.is_buyer_maker,
        start_ns,
        end_ns,
        interval_ns,
    )
    return {str(name): np.asarray(values) for name, values in result.items()}


def compiler_version() -> str:
    """Return the compiler string embedded in the extension."""

    return str(_replay.compiler_version())
