from datetime import date

from spot_perp_lab.data.archives import (
    MarketType,
    archive_filename,
    archive_url,
    checksum_url,
)


def test_spot_archive_url() -> None:
    day = date(2025, 1, 2)
    expected = (
        "https://data.binance.vision/data/spot/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2025-01-02.zip"
    )
    assert archive_url("https://data.binance.vision/", MarketType.SPOT, "BTCUSDT", day) == expected
    assert archive_filename("BTCUSDT", day) == "BTCUSDT-aggTrades-2025-01-02.zip"
    assert checksum_url(expected) == f"{expected}.CHECKSUM"


def test_usdm_perpetual_archive_url() -> None:
    actual = archive_url(
        "https://data.binance.vision", MarketType.PERPETUAL, "ETHUSDT", date(2025, 1, 3)
    )
    assert "/data/futures/um/daily/aggTrades/ETHUSDT/" in actual
