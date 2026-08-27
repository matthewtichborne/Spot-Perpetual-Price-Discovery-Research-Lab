"""SHA-256 checksum parsing and verification."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class ChecksumError(ValueError):
    """Raised when a checksum sidecar or file digest is invalid."""


def parse_checksum(text: str, expected_filename: str | None = None) -> str:
    """Parse a Binance checksum sidecar and return a lowercase digest."""

    fields = text.strip().split()
    if len(fields) != 2 or not SHA256_PATTERN.fullmatch(fields[0]):
        raise ChecksumError("checksum sidecar must contain a SHA-256 digest and filename")
    if expected_filename is not None and Path(fields[1].lstrip("*")).name != expected_filename:
        raise ChecksumError(
            f"checksum filename mismatch: expected {expected_filename}, got {fields[1]}"
        )
    return fields[0].lower()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_digest: str) -> str:
    """Verify a file digest and return the observed digest."""

    observed = sha256_file(path)
    if observed != expected_digest.lower():
        raise ChecksumError(
            f"SHA-256 mismatch for {path.name}: expected {expected_digest}, got {observed}"
        )
    return observed
