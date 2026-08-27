#!/usr/bin/env sh
set -eu

test -f data/manifests/final-evaluation.json
shasum -a 256 -c data/manifests/final-artifacts.sha256
