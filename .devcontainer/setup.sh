#!/usr/bin/env bash
# Runs once, when the codespace is created: packages, then the two default models so the first load is instant.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --user --no-warn-script-location -q --upgrade pip
# CPU build of PyTorch (the default wheel is the CUDA build: several GB, useless without a GPU).
python -m pip install --user --no-warn-script-location -q torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --user --no-warn-script-location -q -r requirements.txt
# The default models of both pages (about 1 GB each), so a load in class does not wait for the Hub.
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"):
    snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.tiktoken"])
    print("cached", repo, flush=True)
PY
echo "QuantBench setup done."
