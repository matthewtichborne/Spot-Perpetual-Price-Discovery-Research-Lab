import hashlib
import zipfile
from datetime import date
from pathlib import Path

from spot_perp_lab.data.archives import MarketType, archive_filename
from spot_perp_lab.data.download import download_archive


def test_download_is_verified_and_cached(tmp_path: Path) -> None:
    day = date(2025, 1, 2)
    filename = archive_filename("BTCUSDT", day)
    source = tmp_path / "source"
    remote = source / "data" / "spot" / "daily" / "aggTrades" / "BTCUSDT" / filename
    remote.parent.mkdir(parents=True)
    with zipfile.ZipFile(remote, "w") as archive:
        archive.writestr("trades.csv", "1,100,1,1,1,1735776000000000,false,true\n")
    digest = hashlib.sha256(remote.read_bytes()).hexdigest()
    remote.with_name(f"{filename}.CHECKSUM").write_text(f"{digest}  {filename}\n")

    raw_root = tmp_path / "raw"
    first = download_archive(source.as_uri(), raw_root, MarketType.SPOT, "BTCUSDT", day)
    second = download_archive(source.as_uri(), raw_root, MarketType.SPOT, "BTCUSDT", day)
    assert first.downloaded is True
    assert second.downloaded is False
    assert second.sha256 == digest

    second.archive_path.write_bytes(b"damaged cache")
    repaired = download_archive(source.as_uri(), raw_root, MarketType.SPOT, "BTCUSDT", day)
    assert repaired.downloaded is True
    assert repaired.sha256 == digest
