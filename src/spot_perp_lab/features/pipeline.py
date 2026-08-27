"""Configuration-driven Phase 3 feature generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from spot_perp_lab.config import AppConfig
from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.checksums import sha256_file
from spot_perp_lab.data.manifest import canonical_json
from spot_perp_lab.data.normalise import processed_path
from spot_perp_lab.data.pipeline import SealedHoldoutError
from spot_perp_lab.features.generate import generate_feature_frame
from spot_perp_lab.features.reporting import write_descriptive_artifacts
from spot_perp_lab.features.validate import validate_feature_frame


@dataclass(frozen=True)
class FeatureResult:
    output_path: Path
    symbol: str
    day: date
    row_count: int
    predictor_count: int
    label_count: int


def feature_output_path(root: Path, config_name: str, symbol: str, day: date) -> Path:
    """Return a deterministic partitioned feature path."""

    return root / config_name / f"symbol={symbol}" / f"date={day.isoformat()}" / "features.parquet"


def _day_bounds(day: date) -> tuple[int, int]:
    start = int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) * 1_000_000_000
    end = (
        int(datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC).timestamp())
        * 1_000_000_000
    )
    return start, end


def generate_features_config(config: AppConfig) -> list[FeatureResult]:
    """Generate, validate, and report features for an unsealed configuration."""

    if config.research.holdout_status == "sealed":
        raise SealedHoldoutError(f"configuration {config.name!r} is sealed")
    if not {"spot", "perpetual"} <= set(config.data.markets):
        raise ValueError("Phase 3 requires both spot and perpetual markets")

    days = [
        config.data.dates.start + timedelta(days=offset)
        for offset in range((config.data.dates.end - config.data.dates.start).days + 1)
    ]
    results: list[FeatureResult] = []
    report_paths: list[Path] = []
    common_predictors: list[str] | None = None
    common_labels: list[str] | None = None
    for day in days:
        for symbol in config.data.symbols:
            spot_path = processed_path(config.data.processed_dir, MarketType.SPOT, symbol, day)
            perpetual_path = processed_path(
                config.data.processed_dir, MarketType.PERPETUAL, symbol, day
            )
            spot = pl.read_parquet(spot_path, hive_partitioning=False)
            perpetual = pl.read_parquet(perpetual_path, hive_partitioning=False)
            start_ns, end_ns = _day_bounds(day)
            frame, predictors, labels = generate_feature_frame(
                spot, perpetual, start_ns, end_ns, config.features
            )
            validate_feature_frame(frame, predictors, labels)
            frame = frame.with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit(day.isoformat()).alias("date"),
            ).select("symbol", "date", *frame.columns)
            output = feature_output_path(config.features.output_dir, config.name, symbol, day)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.part")
            frame.write_parquet(temporary, compression="zstd", statistics=True)
            temporary.replace(output)
            results.append(
                FeatureResult(output, symbol, day, frame.height, len(predictors), len(labels))
            )
            report_paths.append(output)
            common_predictors = predictors
            common_labels = labels

    if not report_paths or common_predictors is None or common_labels is None:
        raise ValueError("configuration selected no feature partitions")
    write_descriptive_artifacts(report_paths, config.name, common_predictors, common_labels)
    source_manifest_path = config.data.manifest_dir / f"{config.name}.json"
    source_manifest_hash = None
    if source_manifest_path.exists():
        source_manifest_hash = json.loads(source_manifest_path.read_text())["manifest_hash"]
    payload = {
        "config_name": config.name,
        "feature_schema_version": "1",
        "source_manifest_hash": source_manifest_hash,
        "feature_config": config.features.model_dump(mode="json"),
        "predictors": common_predictors,
        "labels": common_labels,
        "files": [
            {
                "symbol": result.symbol,
                "date": result.day.isoformat(),
                "file": result.output_path.as_posix(),
                "sha256": sha256_file(result.output_path),
                "rows": result.row_count,
            }
            for result in results
        ],
    }
    manifest_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    document = {**payload, "manifest_hash": manifest_hash}
    config.data.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.data.manifest_dir / f"{config.name}-features.json"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.part")
    temporary_manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return results
