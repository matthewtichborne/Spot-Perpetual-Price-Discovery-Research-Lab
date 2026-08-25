from typer.testing import CliRunner

from spot_perp_lab.cli import app

runner = CliRunner()


def test_show_config() -> None:
    result = runner.invoke(app, ["show-config", "--config", "configs/smoke.yaml"])
    assert result.exit_code == 0
    assert '"name": "btc-two-day-smoke"' in result.stdout


def test_archive_url() -> None:
    result = runner.invoke(
        app,
        [
            "archive-url",
            "--market",
            "perpetual",
            "--symbol",
            "BTCUSDT",
            "--day",
            "2025-01-02",
            "--checksum",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("BTCUSDT-aggTrades-2025-01-02.zip.CHECKSUM")
