#!/usr/bin/env bash
# Start QuantBench on http://127.0.0.1:8000 with the Python that has the packages (see requirements.txt).
#   ./run.sh                       # PYTHON defaults to .venv/bin/python if it exists, else python3
#   ./run.sh --port 8000 --model Qwen/Qwen2.5-Coder-0.5B-Instruct --quant quanto-int8
#   PYTHON=/path/to/python ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
exec "$PYTHON" server.py "$@"
