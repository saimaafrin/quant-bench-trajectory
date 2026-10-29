#!/usr/bin/env bash
# Start the QuantBench server in the background of a dev container / codespace, once per start.
set -u
cd "$(dirname "$0")/.."
PORT="${QB_PORT:-8000}"
LOG=/tmp/quantbench.log

up() { curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/api/health" 2>/dev/null; }

if up; then
  echo "QuantBench is already running on port ${PORT}."
  exit 0
fi
# setsid + nohup: the server must outlive the postStartCommand shell that launched it.
setsid nohup python server.py --port "${PORT}" >"${LOG}" 2>&1 < /dev/null &
for _ in $(seq 1 120); do
  if up; then
    echo "QuantBench is running on port ${PORT} (log: ${LOG})."
    exit 0
  fi
  sleep 1
done
echo "QuantBench did not answer on port ${PORT} within 120 s. Last lines of ${LOG}:" >&2
tail -n 20 "${LOG}" >&2
exit 1
