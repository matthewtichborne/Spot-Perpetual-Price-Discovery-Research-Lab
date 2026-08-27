"""Credential-free, checksum-aware archive downloads with local caching."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

from spot_perp_lab.data.archives import (
    MarketType,
    archive_filename,
    archive_url,
    checksum_url,
)
from spot_perp_lab.data.checksums import ChecksumError, parse_checksum, verify_checksum


@dataclass(frozen=True)
class DownloadResult:
    archive_path: Path
    checksum_path: Path
    sha256: str
    downloaded: bool


def raw_archive_path(root: Path, market: MarketType, symbol: str, day: date) -> Path:
    """Return the deterministic local path for a raw archive."""

    return root / f"market_type={market.value}" / f"symbol={symbol}" / archive_filename(symbol, day)


def _download(url: str, destination: Path) -> None:
    """Stream a URL to an atomic temporary file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    request = Request(url, headers={"User-Agent": "spot-perp-lab/0.1"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download_archive(
    base_url: str,
    raw_root: Path,
    market: MarketType,
    symbol: str,
    day: date,
) -> DownloadResult:
    """Download and verify one archive, or reuse a verified cached copy."""

    remote_archive = archive_url(base_url, market, symbol, day)
    local_archive = raw_archive_path(raw_root, market, symbol, day)
    local_checksum = local_archive.with_name(f"{local_archive.name}.CHECKSUM")

    if local_archive.exists() and local_checksum.exists():
        expected = parse_checksum(local_checksum.read_text(), local_archive.name)
        try:
            observed = verify_checksum(local_archive, expected)
        except ChecksumError:
            local_archive.unlink()
        else:
            return DownloadResult(local_archive, local_checksum, observed, downloaded=False)

    _download(checksum_url(remote_archive), local_checksum)
    expected = parse_checksum(local_checksum.read_text(), local_archive.name)
    _download(remote_archive, local_archive)
    try:
        observed = verify_checksum(local_archive, expected)
    except Exception:
        local_archive.unlink(missing_ok=True)
        raise
    return DownloadResult(local_archive, local_checksum, observed, downloaded=True)
