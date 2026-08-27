"""Explicit spot/futures parsing and canonical Parquet normalisation."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl

from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.schemas import (
    PERPETUAL_COLUMNS,
    SPOT_COLUMNS,
    TimestampUnit,
    timestamp_unit,
)

SCHEMA_VERSION = "1"


class SchemaError(ValueError):
    """Raised when a raw archive does not match its documented source schema."""


@dataclass(frozen=True)
class NormaliseResult:
    output_path: Path
    row_count: int
    timestamp_unit: TimestampUnit


def processed_path(root: Path, market: MarketType, symbol: str, day: date) -> Path:
    """Return the canonical Hive-partitioned Parquet path."""

    return (
        root
        / f"symbol={symbol}"
        / f"market_type={market.value}"
        / f"date={day.isoformat()}"
        / "trades.parquet"
    )


def _read_csv(archive_path: Path, market: MarketType) -> pl.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise SchemaError(
                f"expected exactly one CSV in {archive_path.name}, found {csv_members}"
            )
        payload = archive.read(csv_members[0])

    if market is MarketType.SPOT:
        frame = pl.read_csv(
            io.BytesIO(payload),
            has_header=False,
            new_columns=list(SPOT_COLUMNS),
            infer_schema=False,
        )
    else:
        header = payload.splitlines()[0].decode("utf-8").split(",")
        expected_header = [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ]
        if header != expected_header:
            raise SchemaError(f"unexpected USD-M futures header: {header}")
        frame = pl.read_csv(io.BytesIO(payload), has_header=True, infer_schema=False).rename(
            {"agg_trade_id": "aggregate_trade_id"}
        )
    expected_columns = set(SPOT_COLUMNS if market is MarketType.SPOT else PERPETUAL_COLUMNS)
    if set(frame.columns) != expected_columns:
        raise SchemaError(f"unexpected columns in {archive_path.name}: {frame.columns}")
    return frame


def normalise_archive(
    archive_path: Path,
    processed_root: Path,
    market: MarketType,
    symbol: str,
    day: date,
) -> NormaliseResult:
    """Parse one verified ZIP directly and atomically write canonical Parquet."""

    raw = _read_csv(archive_path, market)
    if raw.height == 0:
        raise SchemaError(f"archive contains no trades: {archive_path.name}")
    maker_values = set(
        raw.select(pl.col("is_buyer_maker").str.to_lowercase().unique()).to_series().to_list()
    )
    if not maker_values <= {"true", "false"}:
        raise SchemaError(f"invalid is_buyer_maker values: {sorted(maker_values)}")

    typed = raw.with_columns(
        pl.col("aggregate_trade_id").cast(pl.Int64, strict=True),
        pl.col("first_trade_id").cast(pl.Int64, strict=True),
        pl.col("last_trade_id").cast(pl.Int64, strict=True),
        pl.col("price").cast(pl.Float64, strict=True),
        pl.col("quantity").cast(pl.Float64, strict=True),
        pl.col("transact_time").cast(pl.Int64, strict=True),
        pl.col("is_buyer_maker").str.to_lowercase().eq("true"),
    )
    minimum_time = cast(int | None, typed["transact_time"].min())
    maximum_time = cast(int | None, typed["transact_time"].max())
    if minimum_time is None or maximum_time is None:
        raise SchemaError("timestamp column contains no values")
    minimum_unit = timestamp_unit(int(minimum_time))
    maximum_unit = timestamp_unit(int(maximum_time))
    if minimum_unit is not maximum_unit:
        raise SchemaError(f"mixed timestamp units: {minimum_unit} and {maximum_unit}")
    multiplier = {
        TimestampUnit.MILLISECOND: 1_000_000,
        TimestampUnit.MICROSECOND: 1_000,
        TimestampUnit.NANOSECOND: 1,
    }[minimum_unit]
    first_event_ns = minimum_time * multiplier
    last_event_ns = maximum_time * multiplier
    partition_start = (
        int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 1_000_000_000
    )
    partition_end = (
        int(datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC).timestamp())
        * 1_000_000_000
    )
    if first_event_ns < partition_start or last_event_ns >= partition_end:
        raise SchemaError(
            f"timestamps fall outside UTC partition {day}: {first_event_ns}..{last_event_ns}"
        )

    normalised = (
        typed.with_columns(
            (pl.col("transact_time") * multiplier).alias("event_time_ns"),
            (pl.col("price") * pl.col("quantity")).alias("notional"),
            pl.when(pl.col("is_buyer_maker"))
            .then(-1)
            .otherwise(1)
            .cast(pl.Int8)
            .alias("aggressor_sign"),
        )
        .with_columns(
            (pl.col("aggressor_sign") * pl.col("quantity")).alias("signed_quantity"),
            (pl.col("aggressor_sign") * pl.col("notional")).alias("signed_notional"),
        )
        .select(
            pl.lit("binance").alias("exchange"),
            pl.lit(market.value).cast(pl.Categorical).alias("market_type"),
            pl.lit(symbol).cast(pl.Categorical).alias("symbol"),
            pl.col("event_time_ns").cast(pl.Int64),
            pl.col("aggregate_trade_id"),
            pl.col("first_trade_id"),
            pl.col("last_trade_id"),
            pl.col("price"),
            pl.col("quantity"),
            pl.col("notional"),
            pl.col("is_buyer_maker"),
            pl.col("aggressor_sign"),
            pl.col("signed_quantity"),
            pl.col("signed_notional"),
            pl.lit(archive_path.name).alias("source_file"),
        )
        .sort(["event_time_ns", "aggregate_trade_id"])
    )

    output = processed_path(processed_root, market, symbol, day)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part")
    normalised.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(output)
    return NormaliseResult(output, normalised.height, minimum_unit)
