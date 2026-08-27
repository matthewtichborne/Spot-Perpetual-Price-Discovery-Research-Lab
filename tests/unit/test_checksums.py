import hashlib
from pathlib import Path

import pytest

from spot_perp_lab.data.checksums import ChecksumError, parse_checksum, verify_checksum


def test_parse_and_verify_checksum(tmp_path: Path) -> None:
    target = tmp_path / "sample.zip"
    target.write_bytes(b"verified bytes")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert parse_checksum(f"{digest}  sample.zip\n", "sample.zip") == digest
    assert verify_checksum(target, digest) == digest


def test_corruption_is_detected(tmp_path: Path) -> None:
    target = tmp_path / "sample.zip"
    target.write_bytes(b"corrupted")
    with pytest.raises(ChecksumError, match="SHA-256 mismatch"):
        verify_checksum(target, "0" * 64)


def test_sidecar_filename_is_validated() -> None:
    with pytest.raises(ChecksumError, match="filename mismatch"):
        parse_checksum(f"{'a' * 64}  wrong.zip", "expected.zip")
