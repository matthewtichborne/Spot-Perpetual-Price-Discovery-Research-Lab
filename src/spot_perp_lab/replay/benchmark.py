"""Reproducible equivalent-work benchmark for Python and C++ replay kernels."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from spot_perp_lab.data.checksums import sha256_file
from spot_perp_lab.data.manifest import canonical_json
from spot_perp_lab.replay.cpp import compiler_version, replay_cpp
from spot_perp_lab.replay.reference import replay_reference, synthetic_market_events

ReplayOutput = dict[str, np.ndarray[Any, Any]]


def _assert_parity(expected: ReplayOutput, actual: ReplayOutput) -> None:
    if expected.keys() != actual.keys():
        raise AssertionError("Python/C++ benchmark output columns differ")
    for name in expected:
        if np.issubdtype(expected[name].dtype, np.integer):
            np.testing.assert_array_equal(actual[name], expected[name])
        else:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-12, atol=1e-12, equal_nan=True
            )


def _time_calls(call: Callable[[], ReplayOutput], repeats: int) -> list[float]:
    call()
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return durations


def run_benchmark(
    *,
    events_per_market: int,
    repeats: int,
    report_path: Path,
    summary_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run the fixed replay benchmark, write reports, and return headline metrics."""

    if events_per_market <= 0 or repeats <= 0:
        raise ValueError("benchmark events and repeats must be positive")
    start_ns = 1_735_776_000_000_000_000
    end_ns = start_ns + 3_600_000_000_000
    interval_ns = 100_000_000
    spot = synthetic_market_events(events_per_market, start_ns, end_ns, 17, 100_000.0)
    perpetual = synthetic_market_events(events_per_market, start_ns, end_ns, 23, 100_010.0)

    def python_call() -> ReplayOutput:
        return replay_reference(spot, perpetual, start_ns, end_ns, interval_ns)

    def cpp_call() -> ReplayOutput:
        return replay_cpp(spot, perpetual, start_ns, end_ns, interval_ns)

    _assert_parity(python_call(), cpp_call())
    python_seconds = _time_calls(python_call, repeats)
    cpp_seconds = _time_calls(cpp_call, repeats)
    python_median = float(np.median(python_seconds))
    cpp_median = float(np.median(cpp_seconds))
    total_events = 2 * events_per_market
    result: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "method": {
            "events_per_market": events_per_market,
            "total_events": total_events,
            "spot_seed": 17,
            "perpetual_seed": 23,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "interval_ns": interval_ns,
            "grid_rows": (end_ns - start_ns) // interval_ns,
            "warmup_calls": 1,
            "timed_repeats": repeats,
            "build_type": "Release",
            "parity_rtol": 1e-12,
            "parity_atol": 1e-12,
        },
        "environment": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "compiler": compiler_version(),
        },
        "python": {
            "seconds": python_seconds,
            "median_seconds": python_median,
            "events_per_second": total_events / python_median,
        },
        "cpp": {
            "seconds": cpp_seconds,
            "median_seconds": cpp_median,
            "events_per_second": total_events / cpp_median,
        },
        "speedup": python_median / cpp_median,
        "scope": (
            "bounded equivalent-work two-market merge and fixed-grid aggregation; "
            "not Polars or end-to-end pipeline speed-up"
        ),
        "parity": "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(
        "# Phase 7 C++ replay benchmark\n\n"
        f"- Events: {total_events:,} total ({events_per_market:,} per market)\n"
        f"- Grid rows: {result['method']['grid_rows']:,}\n"
        f"- Python median: {python_median:.6f} s "
        f"({result['python']['events_per_second']:,.0f} events/s)\n"
        f"- C++ median: {cpp_median:.6f} s "
        f"({result['cpp']['events_per_second']:,.0f} events/s)\n"
        f"- Bounded-kernel speed-up: {result['speedup']:.2f}x\n"
        f"- Parity: {result['parity']} at rtol/atol 1e-12\n"
        f"- Compiler: `{result['environment']['compiler']}`\n\n"
        "This compares identical in-memory merge and fixed-grid aggregation work. It "
        "is not a comparison with Polars and does not represent end-to-end research "
        "pipeline speed-up.\n",
        encoding="utf-8",
    )
    source_paths = {
        "python_reference": Path("src/spot_perp_lab/replay/reference.py"),
        "python_wrapper": Path("src/spot_perp_lab/replay/cpp.py"),
        "cpp_header": Path("cpp/include/spot_perp_lab/replay.hpp"),
        "cpp_core": Path("cpp/src/replay.cpp"),
        "cpp_bindings": Path("cpp/bindings/replay_bindings.cpp"),
        "cmake": Path("CMakeLists.txt"),
        "protocol": Path("docs/phase7_protocol.md"),
        "profile": Path("reports/development/phase7_python_profile.txt"),
    }
    payload = {
        "phase": 7,
        "benchmark": sha256_file(report_path),
        "summary": sha256_file(summary_path),
        "sources": {name: sha256_file(path) for name, path in source_paths.items()},
        "environment": result["environment"],
        "method": result["method"],
        "speedup": result["speedup"],
        "parity": result["parity"],
        "scope": result["scope"],
    }
    manifest_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({**payload, "manifest_hash": manifest_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "events": total_events,
        "python_seconds": python_median,
        "cpp_seconds": cpp_median,
        "speedup": result["speedup"],
        "manifest_hash": manifest_hash,
    }
