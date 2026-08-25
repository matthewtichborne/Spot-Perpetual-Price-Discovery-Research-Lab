from pathlib import Path

import pytest
from pydantic import ValidationError

from spot_perp_lab.config import AppConfig, load_config


def test_smoke_config_loads() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    assert config.name == "btc-two-day-smoke"
    assert config.data.symbols == ("BTCUSDT",)
    assert config.research.holdout_status == "not_applicable"


def test_unknown_config_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AppConfig.model_validate(
            {
                "name": "invalid",
                "data": {
                    "symbols": ["BTCUSDT"],
                    "dates": {"start": "2025-01-02", "end": "2025-01-03"},
                    "typo": True,
                },
            }
        )


def test_reversed_dates_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        AppConfig.model_validate(
            {
                "name": "invalid",
                "data": {
                    "symbols": ["BTCUSDT"],
                    "dates": {"start": "2025-01-03", "end": "2025-01-02"},
                },
            }
        )
