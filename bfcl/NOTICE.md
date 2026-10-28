# Third-party code and data: Berkeley Function Calling Leaderboard (BFCL)

`envs/` and `data/` come from the `bfcl-eval` package, version 2026.3.23
(https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard),
by the Gorilla / Sky Computing Lab team at UC Berkeley, licensed under Apache-2.0 (`LICENSE`).

What was taken and what changed:

- `envs/*.py`: the executable multi-turn tool environments (`func_source_code/`). The only
  changes: imports of `long_context` made relative, and an unused `import subprocess` removed.
- `data/questions.jsonl`, `data/answers.jsonl`: the `multi_turn_base` category (200 tasks)
  and its ground-truth call sequences, copied unchanged.
- `data/func_doc/*.json`: the tool descriptions for each environment, copied unchanged.
- `data/single/*.jsonl`: the single-turn categories simple_python, multiple, parallel,
  parallel_multiple and irrelevance (1,240 tasks) and their possible answers, copied unchanged.
- `ast_checker.py`: BFCL's single-turn AST checker (`eval_checker/ast_eval/ast_checker.py`). Changes:
  the Java/JavaScript converter imports are replaced by stubs and model-specific function-name
  conversion is disabled; the Python checking logic is unchanged.

The multi-turn grading in `../agent_bench.py` reimplements BFCL's multi-turn checker (per-turn state
check plus execution-response check) without `eval()`: model calls are dispatched as JSON
to the environments' public methods and ground-truth calls are parsed with `ast`.

Please cite BFCL when reporting results:
Patil et al., "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic
Evaluation of Large Language Models", ICML 2025.
