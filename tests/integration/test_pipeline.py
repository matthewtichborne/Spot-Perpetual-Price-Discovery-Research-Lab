import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

from spot_perp_lab.config import AppConfig
from spot_perp_lab.data.archives import MarketType
from spot_perp_lab.data.checksums import sha256_file
from spot_perp_lab.data.download import raw_archive_path
from spot_perp_lab.data.pipeline import (
    SealedHoldoutError,
    download_config,
    normalise_config,
    validate_config,
)


def _raw_archive(
    root: Path, market: MarketType, content: str, day: date = date(2025, 1, 2)
) -> None:
    path = raw_archive_path(root, market, "BTCUSDT", day)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trades.csv", content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.CHECKSUM").write_text(f"{digest}  {path.name}\n")


def test_pipeline_is_idempotent_and_leaves_no_extracted_csv(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _raw_archive(
        raw_root,
        MarketType.SPOT,
        "1,100.0,0.25,10,10,1735776000001000,False,True\n",
    )
    _raw_archive(
        raw_root,
        MarketType.PERPETUAL,
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,100.1,0.2,20,20,1735776000002,true\n",
    )
    config = AppConfig.model_validate(
        {
            "name": "integration-smoke",
            "data": {
                "symbols": ["BTCUSDT"],
                "markets": ["spot", "perpetual"],
                "dates": {"start": "2025-01-02", "end": "2025-01-02"},
                "raw_dir": raw_root,
                "processed_dir": tmp_path / "processed",
                "manifest_dir": tmp_path / "manifests",
            },
            "research": {"holdout_status": "not_applicable"},
        }
    )

    first_results, first_hash = normalise_config(config)
    first_parquet_hashes = [sha256_file(item.output_path) for item in first_results]
    second_results, second_hash = normalise_config(config)
    second_parquet_hashes = [sha256_file(item.output_path) for item in second_results]

    assert first_hash == second_hash
    assert first_parquet_hashes == second_parquet_hashes
    assert sum(item.row_count for item in validate_config(config)) == 2
    assert not list(raw_root.rglob("*.csv"))

    manifest = json.loads((tmp_path / "manifests" / "integration-smoke.json").read_text())
    assert manifest["manifest_hash"] == first_hash
    assert len(manifest["files"]) == 2


def test_sealed_holdout_cannot_be_downloaded(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "name": "sealed",
            "data": {
                "symbols": ["BTCUSDT"],
                "dates": {"start": "2025-01-02", "end": "2025-01-02"},
                "raw_dir": tmp_path,
            },
            "research": {"holdout_status": "sealed"},
        }
    )
    with pytest.raises(SealedHoldoutError, match="is sealed"):
        download_config(config)
