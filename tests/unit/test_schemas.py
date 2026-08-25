import pytest

from spot_perp_lab.data.schemas import TimestampUnit, timestamp_unit, to_nanoseconds


def test_sampled_spot_timestamp_is_microseconds() -> None:
    assert timestamp_unit(1_735_776_000_113_701) is TimestampUnit.MICROSECOND
    assert to_nanoseconds(1_735_776_000_113_701) == 1_735_776_000_113_701_000


def test_sampled_perpetual_timestamp_is_milliseconds() -> None:
    assert timestamp_unit(1_735_776_005_115) is TimestampUnit.MILLISECOND
    assert to_nanoseconds(1_735_776_005_115) == 1_735_776_005_115_000_000


def test_unknown_timestamp_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported timestamp scale"):
        timestamp_unit(12345)
