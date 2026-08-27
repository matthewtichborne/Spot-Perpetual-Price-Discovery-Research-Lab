import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.normalise import SchemaError, normalise_archive
from spot_perp_lab.data.schemas import TimestampUnit
from spot_perp_lab.data.validate import DataValidationError, validate_parquet


def _zip(path: Path, content: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.with_suffix(".csv").name, content)


def test_spot_normalisation_and_validation(tmp_path: Path) -> None:
    source = tmp_path / "BTCUSDT-aggTrades-2025-01-02.zip"
    _zip(
        source,
        "2,101.0,0.5,11,11,1735776000002000,True,True\n"
        "1,100.0,0.25,10,10,1735776000001000,False,True\n",
    )
    result = normalise_archive(
        source, tmp_path / "processed", MarketType.SPOT, "BTCUSDT", date(2025, 1, 2)
    )
    frame = pl.read_parquet(result.output_path)
    assert result.timestamp_unit is TimestampUnit.MICROSECOND
    assert frame["event_time_ns"].to_list() == [
        1_735_776_000_001_000_000,
        1_735_776_000_002_000_000,
    ]
    assert frame["aggressor_sign"].to_list() == [1, -1]
    assert frame["signed_quantity"].to_list() == [0.25, -0.5]
    assert validate_parquet(result.output_path, "spot", "BTCUSDT").row_count == 2


def test_futures_header_and_millisecond_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "BTCUSDT-aggTrades-2025-01-02.zip"
    _zip(
        source,
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,100.0,0.1,10,10,1735776000001,true\n",
    )
    result = normalise_archive(
        source,
        tmp_path / "processed",
        MarketType.PERPETUAL,
        "BTCUSDT",
        date(2025, 1, 2),
    )
    assert result.timestamp_unit is TimestampUnit.MILLISECOND
    assert pl.read_parquet(result.output_path)["event_time_ns"][0] == 1_735_776_000_001_000_000


def test_bad_futures_header_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.zip"
    _zip(source, "wrong,header\n1,2\n")
    with pytest.raises(SchemaError, match="unexpected USD-M futures header"):
        normalise_archive(
            source,
            tmp_path / "processed",
            MarketType.PERPETUAL,
            "BTCUSDT",
            date(2025, 1, 2),
        )


def test_invalid_maker_flag_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad-maker.zip"
    _zip(source, "1,100.0,0.25,10,10,1735776000001000,unknown,True\n")
    with pytest.raises(SchemaError, match="invalid is_buyer_maker"):
        normalise_archive(
            source, tmp_path / "processed", MarketType.SPOT, "BTCUSDT", date(2025, 1, 2)
        )


def test_timestamp_outside_partition_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "wrong-day.zip"
    _zip(source, "1,100.0,0.25,10,10,1735862400001000,False,True\n")
    with pytest.raises(SchemaError, match="outside UTC partition"):
        normalise_archive(
            source, tmp_path / "processed", MarketType.SPOT, "BTCUSDT", date(2025, 1, 2)
        )


def test_corrupted_parquet_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "BTCUSDT-aggTrades-2025-01-02.zip"
    _zip(source, "1,100.0,0.25,10,10,1735776000001000,False,True\n")
    result = normalise_archive(
        source, tmp_path / "processed", MarketType.SPOT, "BTCUSDT", date(2025, 1, 2)
    )
    frame = pl.read_parquet(result.output_path).with_columns(pl.lit(-1.0).alias("price"))
    frame.write_parquet(result.output_path)
    with pytest.raises(DataValidationError, match="invalid_numeric"):
        validate_parquet(result.output_path, "spot", "BTCUSDT")
