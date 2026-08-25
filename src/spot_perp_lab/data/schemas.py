"""Source schema facts established during the feasibility spike."""

from enum import StrEnum

SPOT_COLUMNS = (
    "aggregate_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
)

PERPETUAL_COLUMNS = (
    "aggregate_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)


class TimestampUnit(StrEnum):
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"


def timestamp_unit(value: int) -> TimestampUnit:
    """Classify a contemporary Unix timestamp by its decimal scale.

    Values outside plausible modern millisecond-to-nanosecond scales are rejected
    instead of silently assigned a unit.
    """

    digits = len(str(abs(value)))
    if digits == 13:
        return TimestampUnit.MILLISECOND
    if digits == 16:
        return TimestampUnit.MICROSECOND
    if digits == 19:
        return TimestampUnit.NANOSECOND
    raise ValueError(f"unsupported timestamp scale ({digits} digits): {value}")


def to_nanoseconds(value: int) -> int:
    """Convert an explicitly detected contemporary Unix timestamp to nanoseconds."""

    unit = timestamp_unit(value)
    multiplier = {
        TimestampUnit.MILLISECOND: 1_000_000,
        TimestampUnit.MICROSECOND: 1_000,
        TimestampUnit.NANOSECOND: 1,
    }[unit]
    return value * multiplier
