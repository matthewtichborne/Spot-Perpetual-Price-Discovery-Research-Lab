"""Deterministic processed-data manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spot_perp_lab.data.normalise import SCHEMA_VERSION


@dataclass(frozen=True)
class ManifestRecord:
    market_type: str
    symbol: str
    date: str
    raw_file: str
    raw_sha256: str
    parquet_file: str
    parquet_sha256: str
    row_count: int
    timestamp_unit: str


def canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically for hashing and version control."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def manifest_payload(config_name: str, records: Iterable[ManifestRecord]) -> dict[str, Any]:
    """Build the stable portion of a manifest."""

    ordered = sorted(
        (asdict(record) for record in records),
        key=lambda item: (item["date"], item["symbol"], item["market_type"]),
    )
    return {"config_name": config_name, "schema_version": SCHEMA_VERSION, "files": ordered}


def write_manifest(
    manifest_dir: Path, config_name: str, records: Iterable[ManifestRecord]
) -> tuple[Path, str]:
    """Atomically write a manifest and return its stable content hash."""

    payload = manifest_payload(config_name, records)
    content_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    document = {**payload, "manifest_hash": content_hash}
    manifest_dir.mkdir(parents=True, exist_ok=True)
    destination = manifest_dir / f"{config_name}.json"
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination, content_hash
