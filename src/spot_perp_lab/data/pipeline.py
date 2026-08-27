"""Configuration-driven Phase 2 data pipeline orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from spot_perp_lab.config import AppConfig
from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.checksums import parse_checksum, sha256_file, verify_checksum
from spot_perp_lab.data.download import DownloadResult, download_archive, raw_archive_path
from spot_perp_lab.data.manifest import ManifestRecord, write_manifest
from spot_perp_lab.data.normalise import NormaliseResult, normalise_archive, processed_path
from spot_perp_lab.data.validate import ValidationResult, validate_parquet


class SealedHoldoutError(ValueError):
    """Raised when a data command targets a sealed configuration."""


def _ensure_unsealed(config: AppConfig) -> None:
    if config.research.holdout_status == "sealed":
        raise SealedHoldoutError(
            f"configuration {config.name!r} is sealed; freeze the research design before access"
        )


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _selections(config: AppConfig) -> list[tuple[MarketType, str, date]]:
    return [
        (MarketType(market), symbol, day)
        for day in _days(config.data.dates.start, config.data.dates.end)
        for symbol in config.data.symbols
        for market in config.data.markets
    ]


def download_config(config: AppConfig) -> list[DownloadResult]:
    """Download every raw archive selected by an unsealed configuration."""

    _ensure_unsealed(config)
    selections = _selections(config)

    def download(selection: tuple[MarketType, str, date]) -> DownloadResult:
        market, symbol, day = selection
        return download_archive(str(config.data.base_url), config.data.raw_dir, market, symbol, day)

    with ThreadPoolExecutor(max_workers=min(4, len(selections))) as executor:
        return list(executor.map(download, selections))


def normalise_config(config: AppConfig) -> tuple[list[NormaliseResult], str]:
    """Verify and normalise all configured raw files, then write a stable manifest."""

    _ensure_unsealed(config)
    results: list[NormaliseResult] = []
    records: list[ManifestRecord] = []
    for market, symbol, day in _selections(config):
        archive = raw_archive_path(config.data.raw_dir, market, symbol, day)
        sidecar = archive.with_name(f"{archive.name}.CHECKSUM")
        if not archive.exists() or not sidecar.exists():
            raise FileNotFoundError(f"download is incomplete for {market.value} {symbol} {day}")
        expected = parse_checksum(sidecar.read_text(), archive.name)
        raw_digest = verify_checksum(archive, expected)
        result = normalise_archive(archive, config.data.processed_dir, market, symbol, day)
        results.append(result)
        records.append(
            ManifestRecord(
                market_type=market.value,
                symbol=symbol,
                date=day.isoformat(),
                raw_file=archive.as_posix(),
                raw_sha256=raw_digest,
                parquet_file=result.output_path.as_posix(),
                parquet_sha256=sha256_file(result.output_path),
                row_count=result.row_count,
                timestamp_unit=result.timestamp_unit.value,
            )
        )
    _, manifest_hash = write_manifest(config.data.manifest_dir, config.name, records)
    return results, manifest_hash


def validate_config(config: AppConfig) -> list[ValidationResult]:
    """Run canonical DuckDB validations for every configured partition."""

    _ensure_unsealed(config)
    return [
        validate_parquet(
            processed_path(config.data.processed_dir, market, symbol, day), market.value, symbol
        )
        for market, symbol, day in _selections(config)
    ]
