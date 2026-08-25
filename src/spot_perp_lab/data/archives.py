"""Pure URL and filename construction for official Binance archives."""

from datetime import date
from enum import StrEnum


class MarketType(StrEnum):
    SPOT = "spot"
    PERPETUAL = "perpetual"


def archive_filename(symbol: str, day: date) -> str:
    """Return Binance's daily aggregate-trade archive filename."""

    return f"{symbol}-aggTrades-{day.isoformat()}.zip"


def archive_url(base_url: str, market: MarketType, symbol: str, day: date) -> str:
    """Build the official daily aggregate-trade archive URL."""

    root = base_url.rstrip("/")
    market_path = "spot" if market is MarketType.SPOT else "futures/um"
    filename = archive_filename(symbol, day)
    return f"{root}/data/{market_path}/daily/aggTrades/{symbol}/{filename}"


def checksum_url(archive: str) -> str:
    """Return the checksum sidecar URL for an archive URL."""

    return f"{archive}.CHECKSUM"
