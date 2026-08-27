#!/usr/bin/env sh
set -eu

config_path=${1:-configs/smoke.yaml}

python -m spot_perp_lab.cli show-config --config "$config_path"
python -m spot_perp_lab.cli download --config "$config_path"
python -m spot_perp_lab.cli normalise --config "$config_path"
python -m spot_perp_lab.cli validate --config "$config_path"
python -m spot_perp_lab.cli features --config "$config_path"
