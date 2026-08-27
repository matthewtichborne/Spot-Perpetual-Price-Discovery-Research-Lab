"""Command-line entry point for the research pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from spot_perp_lab.config import load_config
from spot_perp_lab.data.archives import MarketType, archive_url, checksum_url
from spot_perp_lab.data.pipeline import download_config, normalise_config, validate_config
from spot_perp_lab.execution.config import load_phase6_config
from spot_perp_lab.execution.pipeline import run_phase6
from spot_perp_lab.features.pipeline import generate_features_config
from spot_perp_lab.replay.benchmark import run_benchmark
from spot_perp_lab.research.phase5 import run_phase5
from spot_perp_lab.research.phase8 import run_phase8
from spot_perp_lab.research.pipeline import run_phase4

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


@app.command("download")
def download(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Download and checksum-verify configured raw archives."""

    results = download_config(load_config(config))
    downloaded = sum(result.downloaded for result in results)
    typer.echo(
        f"verified={len(results)} downloaded={downloaded} cached={len(results) - downloaded}"
    )


@app.command("normalise")
def normalise(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Convert verified raw ZIPs directly to canonical partitioned Parquet."""

    results, manifest_hash = normalise_config(load_config(config))
    rows = sum(result.row_count for result in results)
    typer.echo(f"partitions={len(results)} rows={rows} manifest_hash={manifest_hash}")


@app.command("validate")
def validate(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate configured Parquet partitions using DuckDB."""

    results = validate_config(load_config(config))
    rows = sum(result.row_count for result in results)
    typer.echo(f"partitions={len(results)} rows={rows} status=ok")


@app.command("features")
def features(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Generate leakage-safe fixed-grid features and future-return labels."""

    results = generate_features_config(load_config(config))
    rows = sum(result.row_count for result in results)
    typer.echo(
        f"partitions={len(results)} rows={rows} "
        f"predictors={results[0].predictor_count} labels={results[0].label_count}"
    )


@app.command("train")
def train(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run the frozen Phase 4 walk-forward research procedure."""

    result = run_phase4(load_config(config))
    typer.echo(
        f"folds={result['folds']} rows={result['rows']} failures={result['failures']} "
        f"preferred={result['preferred_model']}/{result['preferred_scope']} "
        f"manifest_hash={result['manifest_hash']}"
    )


@app.command("confirm")
def confirm(
    development_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    confirmation_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run the frozen Phase 5 confirmation and robustness procedure."""

    result = run_phase5(load_config(development_config), load_config(confirmation_config))
    typer.echo(
        f"selected={result['selected_model']}/expanded "
        f"xgboost={result['selected_xgboost']} failures={result['failures']} "
        f"manifest_hash={result['manifest_hash']}"
    )


@app.command("backtest")
def backtest(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run the frozen Phase 6 execution and portfolio sensitivity analysis."""

    result = run_phase6(load_phase6_config(config))
    typer.echo(
        f"trades={result['primary_trades']} net_pnl={result['primary_net_pnl']:.6g} "
        f"conclusion={result['conclusion']} manifest_hash={result['manifest_hash']}"
    )


@app.command("benchmark-replay")
def benchmark_replay(
    events_per_market: Annotated[int, typer.Option(min=1)] = 1_000_000,
    repeats: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    """Benchmark equivalent Python/C++ two-market replay work."""

    result = run_benchmark(
        events_per_market=events_per_market,
        repeats=repeats,
        report_path=Path("reports/development/phase7_benchmark.json"),
        summary_path=Path("reports/development/phase7_benchmark.md"),
        manifest_path=Path("data/manifests/phase7-replay.json"),
    )
    typer.echo(
        f"events={result['events']} python={result['python_seconds']:.6f}s "
        f"cpp={result['cpp_seconds']:.6f}s speedup={result['speedup']:.2f}x "
        f"manifest_hash={result['manifest_hash']}"
    )


@app.command("final-evaluate")
def final_evaluate(
    development_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    confirmation_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    final_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    phase6_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run the registered one-time Phase 8 final-holdout evaluation."""

    result = run_phase8(
        load_config(development_config),
        load_config(confirmation_config),
        load_config(final_config),
        load_phase6_config(phase6_config),
    )
    typer.echo(
        f"oos_r2={result['oos_r2']:.6g} trades={result['trades']} "
        f"net_pnl={result['net_pnl']:.6g} manifest_hash={result['manifest_hash']}"
    )


if __name__ == "__main__":
    app()
