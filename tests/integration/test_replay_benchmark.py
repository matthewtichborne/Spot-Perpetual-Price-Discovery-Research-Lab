from __future__ import annotations

import json
from pathlib import Path

from spot_perp_lab.replay.benchmark import run_benchmark


def test_small_replay_benchmark_writes_scoped_auditable_report(tmp_path: Path) -> None:
    report = tmp_path / "benchmark.json"
    summary = tmp_path / "benchmark.md"
    manifest = tmp_path / "manifest.json"
    result = run_benchmark(
        events_per_market=5_000,
        repeats=1,
        report_path=report,
        summary_path=summary,
        manifest_path=manifest,
    )
    assert result["events"] == 10_000
    assert result["python_seconds"] > 0
    assert result["cpp_seconds"] > 0
    document = json.loads(report.read_text())
    assert document["parity"] == "passed"
    assert "not Polars" in document["scope"]
    assert json.loads(manifest.read_text())["manifest_hash"] == result["manifest_hash"]
