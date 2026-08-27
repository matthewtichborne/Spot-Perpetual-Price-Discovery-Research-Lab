"""Profile the pure-Python Phase 7 replay reference on a fixed workload."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from pathlib import Path

from spot_perp_lab.replay.reference import replay_reference, synthetic_market_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-per-market", type=int, default=100_000)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/development/phase7_python_profile.txt")
    )
    args = parser.parse_args()
    start_ns = 1_735_776_000_000_000_000
    end_ns = start_ns + 3_600_000_000_000
    spot = synthetic_market_events(args.events_per_market, start_ns, end_ns, 17, 100_000.0)
    perpetual = synthetic_market_events(args.events_per_market, start_ns, end_ns, 23, 100_010.0)
    profiler = cProfile.Profile()
    profiler.enable()
    replay_reference(spot, perpetual, start_ns, end_ns, 100_000_000)
    profiler.disable()
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumulative")
    stats.print_stats(20)
    header = (
        "Phase 7 pre-C++ Python profile\n"
        f"events_per_market={args.events_per_market}\n"
        "interval_ns=100000000\n"
        "window_ns=3600000000000\n\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
