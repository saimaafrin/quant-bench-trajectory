# Third-party data: ToolHop

ToolHop (Ye et al., "ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in
Multi-Hop Tool Use", ACL 2025), by ByteDance Research, licensed under Apache-2.0 (`LICENSE`).
Data: https://huggingface.co/datasets/bytedance-research/ToolHop (`data/ToolHop.json`, 995
queries, 3,912 tools). It is downloaded from the Hub on first use, not stored here.

`../agent_bench.py` follows the official open-model evaluation (`code/evaluation_open.py`,
"Mandatory" scenario): the same user prompt, at most 9 model turns, tool results returned as
JSON, and the same answer check (match inside `<answer>` tags, or the answer appearing in the
last tool output). Differences: `ast.literal_eval` replaces `eval` when comparing answers, and
the dataset's tool functions run in a separate, isolated Python process (`worker.py`) with a
10-second limit per call instead of being `exec`'d in the evaluation process.
