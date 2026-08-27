"""DuckDB-backed validation for canonical trade Parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

REQUIRED_COLUMNS = {
    "exchange",
    "market_type",
    "symbol",
    "event_time_ns",
    "aggregate_trade_id",
    "first_trade_id",
    "last_trade_id",
    "price",
    "quantity",
    "notional",
    "is_buyer_maker",
    "aggressor_sign",
    "signed_quantity",
    "signed_notional",
    "source_file",
}


class DataValidationError(ValueError):
    """Raised when normalised data violates a canonical invariant."""


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    row_count: int
    first_event_time_ns: int
    last_event_time_ns: int


def validate_parquet(path: Path, expected_market: str, expected_symbol: str) -> ValidationResult:
    """Validate schema, ordering, identities, signs, and numeric sanity in DuckDB."""

    connection = duckdb.connect(":memory:")
    try:
        description = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning = false)", [str(path)]
        ).fetchall()
        columns = {str(row[0]) for row in description}
        if columns != REQUIRED_COLUMNS:
            raise DataValidationError(
                f"schema mismatch for {path}: missing={REQUIRED_COLUMNS - columns}, "
                f"extra={columns - REQUIRED_COLUMNS}"
            )

        summary = connection.execute(
            """
            WITH numbered AS (
                SELECT *, row_number() OVER () AS physical_row
                FROM read_parquet(?, hive_partitioning = false)
            ), ordered AS (
                SELECT *, lag(event_time_ns) OVER (ORDER BY physical_row) AS previous_time
                FROM numbered
            )
            SELECT
                count(*) AS row_count,
                min(event_time_ns) AS first_time,
                max(event_time_ns) AS last_time,
                count(*) FILTER (WHERE event_time_ns < previous_time) AS time_reversals,
                count(*) FILTER (
                    WHERE price <= 0 OR quantity <= 0 OR notional <= 0
                       OR NOT isfinite(price) OR NOT isfinite(quantity) OR NOT isfinite(notional)
                ) AS invalid_numeric,
                count(*) FILTER (WHERE aggressor_sign NOT IN (-1, 1)) AS invalid_sign,
                count(*) FILTER (
                    WHERE aggressor_sign != CASE WHEN is_buyer_maker THEN -1 ELSE 1 END
                       OR abs(signed_quantity - aggressor_sign * quantity) > 1e-12
                       OR abs(signed_notional - aggressor_sign * notional)
                          > 1e-12 * greatest(1.0, abs(notional))
                ) AS invalid_signed_flow,
                count(*) FILTER (
                    WHERE exchange != 'binance' OR market_type != ? OR symbol != ?
                ) AS invalid_identity,
                count(*) FILTER (
                    WHERE event_time_ns IS NULL OR aggregate_trade_id IS NULL OR price IS NULL
                       OR quantity IS NULL OR is_buyer_maker IS NULL
                ) AS required_nulls
            FROM ordered
            """,
            [str(path), expected_market, expected_symbol],
        ).fetchone()
        if summary is None:
            raise DataValidationError(f"could not query {path}")
        row_count, first_time, last_time, *violations = (int(value or 0) for value in summary)
        duplicate_summary = connection.execute(
            """
            SELECT count(*) FROM (
                SELECT aggregate_trade_id
                FROM read_parquet(?, hive_partitioning = false)
                GROUP BY aggregate_trade_id
                HAVING count(*) > 1
            )
            """,
            [str(path)],
        ).fetchone()
        if duplicate_summary is None:
            raise DataValidationError(f"could not query duplicate IDs in {path}")
        duplicate_ids = int(duplicate_summary[0])
        if row_count == 0:
            raise DataValidationError(f"empty Parquet file: {path}")
        names = [
            "time_reversals",
            "invalid_numeric",
            "invalid_sign",
            "invalid_signed_flow",
            "invalid_identity",
            "required_nulls",
        ]
        problems = {name: value for name, value in zip(names, violations, strict=True) if value}
        if duplicate_ids:
            problems["duplicate_aggregate_trade_ids"] = duplicate_ids
        if problems:
            raise DataValidationError(f"validation failed for {path}: {problems}")
        return ValidationResult(path, row_count, first_time, last_time)
    finally:
        connection.close()
