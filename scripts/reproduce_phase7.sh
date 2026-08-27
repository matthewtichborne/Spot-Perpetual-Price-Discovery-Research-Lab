#!/usr/bin/env sh
set -eu

python -m pip install -e '.[dev]'
python scripts/profile_replay.py \
  --events-per-market 100000 \
  --output reports/development/phase7_python_profile.txt
spot-perp benchmark-replay --events-per-market 1000000 --repeats 3
pytest -p no:cacheprovider tests/unit/test_replay_reference.py tests/parity/test_replay_parity.py
