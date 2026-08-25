#!/usr/bin/env sh
set -eu

config_path=${1:-configs/smoke.yaml}

python -m spot_perp_lab.cli show-config --config "$config_path"
printf '%s\n' "Phase 1 scaffold verified; full reproduction is implemented in later phases."
