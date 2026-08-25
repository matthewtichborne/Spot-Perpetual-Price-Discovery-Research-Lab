"""Command-line entry point for the research pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from spot_perp_lab.config import load_config
from spot_perp_lab.data.archives import MarketType, archive_url, checksum_url

app = typer.Typer(no_args_is_help=True, help="Spot-perpetual price discovery research tools.")


@app.command("show-config")
def show_config(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a config and print its canonical JSON form."""

    loaded = load_config(config)
    typer.echo(loaded.model_dump_json(indent=2))


@app.command("archive-url")
def show_archive_url(
    market: Annotated[MarketType, typer.Option()],
    symbol: Annotated[str, typer.Option()],
    day: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    checksum: Annotated[bool, typer.Option(help="Print checksum sidecar URL.")] = False,
) -> None:
    """Print a deterministic official daily aggregate-trade URL."""

    url = archive_url("https://data.binance.vision", market, symbol.upper(), day.date())
    typer.echo(checksum_url(url) if checksum else url)


if __name__ == "__main__":
    app()
