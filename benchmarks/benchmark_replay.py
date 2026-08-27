"""Command-line wrapper for the Phase 7 equivalent-work replay benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from spot_perp_lab.replay.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-per-market", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/development/phase7_benchmark.json")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/development/phase7_benchmark.md")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/phase7-replay.json"))
    args = parser.parse_args()
    result = run_benchmark(
        events_per_market=args.events_per_market,
        repeats=args.repeats,
        report_path=args.report,
        summary_path=args.summary,
        manifest_path=args.manifest,
    )
    print(
        f"events={result['events']} python={result['python_seconds']:.6f}s "
        f"cpp={result['cpp_seconds']:.6f}s speedup={result['speedup']:.2f}x "
        f"manifest_hash={result['manifest_hash']}"
    )


if __name__ == "__main__":
    main()
