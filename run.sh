#!/usr/bin/env bash
# Start QuantBench Live. Override the env with PYTHON=/path/to/python ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/scratch/oldhome/safrin/anaconda3_new/envs/bigcodebench-gen/bin/python}"
exec "$PYTHON" server.py "$@"
