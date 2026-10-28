# QuantBench Live

Web dashboard that loads a full-precision LLM and a quantized copy of the same Hugging Face
model side by side, sends both the same coding task (built-in, HumanEval or BigCodeBench), and
streams the answers live. Each answer is then unit-tested and statically analyzed, so you can
compare speed, memory, correctness and code quality.

## Run

```bash
./run.sh                      # uses the bigcodebench-gen conda env, serves on 127.0.0.1:8000
./run.sh --port 8000 --model Qwen/Qwen2.5-Coder-7B-Instruct --quant bnb-8bit
PYTHON=/path/to/python ./run.sh   # another env needs: torch transformers accelerate bitsandbytes
                                  # fastapi uvicorn pandas ruff pylint radon bandit
```

The server runs on the GPU machine. From your laptop, forward the port and open the page:

```bash
ssh -L 8000:localhost:8000 <user>@<gpu-host>
# then browse to http://localhost:8000
```

## What the page shows

1. **Load models**: slot A (baseline) and slot B (quantized). Pick a model from the dropdown
   (open code models up to ~7B, or "Custom repo…"), a precision and a GPU. Before you load, the
   card estimates the weight memory against the GPU's free memory. Loading shows live stages
   (download progress, tokenizer, weights → GPU, warm-up); when it finishes, the card shows
   footprint, load time and size on disk.
2. **Coding task**: choose a **Source**:
   - *Built-in tasks*: 10 small tasks (8 with editable assert tests).
   - *HumanEval* (164): graded with the official `check()` function, one pass/fail per problem.
   - *BigCodeBench-Hard* (148) and *BigCodeBench full* (1140), instruct split v0.1.4: graded with
     each task's `unittest` suite, with every test method reported.

   Benchmarks download from the Hub on first use. For a benchmark source, **Batch evaluation**
   runs N problems in a row and builds a report (pass@1, speed, quality means, per-problem
   table, CSV/JSON export).
3. **Live outputs**: both answers stream with tokens, decode tok/s, TTFT, elapsed time and peak
   memory. Under each answer you get:
   - **Tests**: pass/fail badges (hover for the failing assertion).
   - **Quality**: static analysis of the extracted code:
     - **pylint** score 0–10
     - **ruff** lint issues (pycodestyle, pyflakes, bugbear, simplify, comprehensions, naming)
     - **radon** cyclomatic complexity and maintainability index, plus SLOC
     - **bandit** security findings

     Expand "findings" to see every message with its line number.
4. **Comparison**: memory saving, speed ratio, TTFT, tests, pylint/ruff, output similarity, and a
   tokens-over-time chart.
5. **Run history**: everything you generate is saved on the server in `runs/history.sqlite`, so
   it survives page reloads and server restarts. Each **run** is one row: a batch is one run, and
   a single Generate is a run of one problem. The row shows pass@1, mean pylint, ruff issues per
   problem and tok/s for A and B. Click a run to open its report:
   - **Overall results**, A vs B, with the better value underlined:
     - Correctness: pass@1, individual test cases passed, code that failed to run, timeouts.
     - Code quality: mean pylint score, ruff issues (total and per problem), syntax errors, mean
       complexity, mean maintainability index, bandit findings (with high/medium counts), SLOC.
     - Efficiency: mean decode speed, mean TTFT, tokens, total generation time, weights memory,
       peak memory. Also the mean output similarity between A and B.
   - **Outcomes per problem**: both pass / only A passes (B regressed) / only B passes / both
     fail. Click a box to list those problems.
   - **Issues raised**: every pylint, ruff and bandit code in the run, with how often each model
     triggered it and in how many problems. Filter by tool; click an issue to list the problems
     where it occurs.
   - **Problems**: filter by outcome (A failed, B failed, regressions, didn't run, syntax error,
     disagreements…) or search by name. Click a problem to see both outputs, every failing test
     with its error, every finding with its line number, and the exact code that was tested, or
     to load the task back into the prompt.

   **Export CSV** on the history saves one line per run with all aggregate metrics. **Export
   problems CSV** in a report saves one line per problem. **Delete run** removes a run. Use
   `--db path.sqlite` to keep a separate history per experiment.

## Hosted version (Vercel + Supabase + each user's own GPU)

The same page can be published as a website where people sign in, connect their **own** hardware
(a free Google Colab GPU or their own computer) and run comparisons in parallel. All runs go to one shared
history.

```
browser ──▶ Vercel (static site, deploy/quantbench) ──▶ Supabase (logins + shared run history)
   │
   └──▶ the user's own backend running connect.py (server.py + random access key):
        Colab or a remote GPU server  → reached through a Cloudflare quick tunnel
        the user's own laptop/desktop → reached directly at http://127.0.0.1 (--local)
```

- **Setup (once):**
  1. Create a Supabase project and run `deploy/supabase.sql` in its SQL editor.
  2. In **Authentication → URL Configuration**, set the Site URL and add the Vercel URL to the
     redirect URLs.
  3. Build and deploy:
     `python deploy/build_site.py --supabase-url … --supabase-anon-key … --site https://<site>`,
     then `npx vercel deploy deploy/quantbench --prod --scope <team>`.
- **Access:** only emails in `public.allowed_emails` can use the site. Add `'*'` to that table to
  allow anyone who signs up. Users can delete only their own runs.
- **Connecting a backend:** the connect card has two tabs.
  - **Google Colab:** a two-line cell downloads the backend from the site, starts it with a
    random key, opens a tunnel, and prints an **Open QuantBench** link that carries the backend
    URL and key in the URL fragment (never sent to a server).
  - **My own computer** (macOS, Linux, Windows; Python 3.10+):
    `curl -fsSL https://<site>/backend.tar.gz | tar xz` then
    `python3 quantbench/connect.py --site https://<site> --local`. It installs missing packages,
    starts the backend on 127.0.0.1 (no tunnel) and opens the site already connected. It runs in
    the foreground; Ctrl+C stops it (`--detach` keeps it in the background, `--stop` ends it).
    Chrome may ask to allow access to local network devices. Safari can block an https page from
    calling http://127.0.0.1; use another browser, or drop `--local`.
  - **A remote GPU server** (e.g. ams1 over SSH): the same command without `--local` uses the
    tunnel and prints a link you can open anywhere.
- **Hardware:** with an NVIDIA GPU every option works. On Apple Silicon (`mps`) or CPU-only
  machines bitsandbytes is unavailable (it is greyed out in the UI), and connect.py installs
  optimum-quanto instead: `quanto-int8` / `quanto-int4`. Use 0.5B–1.5B models there. Measured on
  CPU with Qwen2.5-Coder-0.5B-Instruct: FP32 1.98 GB, quanto INT8 0.90 GB, INT4 0.75 GB (the
  embeddings stay in full precision). INT4 on CPU compiles a kernel on first load (needs a C++
  compiler) and decodes much more slowly than INT8, so INT8 is the default there.
- **Limits:**
  - Free Colab has one 16 GB T4: about 3B models in FP16 next to their 4-bit version, or 7B in
    8-bit/4-bit. 7B in FP16 needs a larger GPU.
  - Sessions end when idle; rerun the cell to reconnect.
  - Google's Colab policy restricts web services on free runtimes, so sessions may be stopped.
  - Generated code runs on the user's own backend, never on Vercel or Supabase.
- **Local mode is unchanged:** `./run.sh` serves the same page with no login, history in
  `runs/history.sqlite`.

## Agent page: tool-calling trajectories (`/agent`)

A second page (`static/agent.html`, served at `/agent` locally and on the hosted site) runs the two
loaded models as **tool-calling agents** on the same task and shows their trajectories side by side
with the benchmark's reference trajectory. The code-generation page is unchanged; both pages share
the backend, the loaded models and the connection.

- **Benchmarks** (`agent_benchmarks.py`, chosen in the Task section):
  - **BFCL multi-turn** (Berkeley Function Calling Leaderboard v4, multi-turn base, Apache-2.0): 200
    tasks of 1–7 user turns against stateful, executable tools (file system, messaging, posting,
    tickets, trading, travel booking, vehicle control, math). Vendored in `bfcl/`.
  - **BFCL single-turn** (same suite): 1,240 tasks in five categories: simple, multiple, parallel,
    parallel multiple, and irrelevance (the agent must call nothing). One reply, graded by BFCL's own
    AST checker (`bfcl/ast_checker.py`, Python path only).
  - **ToolHop** (ACL 2025, Apache-2.0): 995 multi-hop questions answered by chaining 3–7 executable
    tools; graded on the final answer like the official evaluation, with at most 9 model turns. The
    data is downloaded from the Hub on first use; its tool functions (dataset code) run in a separate
    isolated process (`toolhop/worker.py`), 10 s per call. See `toolhop/NOTICE.md`.
- **Agent loop** (`agent_runner.py`): per user turn the model may call tools for up to the
  benchmark's step limit (BFCL 20, ToolHop 9, single-turn 1); the turn ends when it replies without a
  tool call. A **loop guard** (default 3, 0 = off) ends a turn once that many of its steps have made
  the identical call (duplicates within one reply count once), so a looping model doesn't burn 20 steps; the run is then labelled "stuck repeating
  a call". BFCL multi-turn uses BFCL's own multi-turn instruction in the system prompt. Models with a tool-aware chat
  template (Qwen2.5, Hammer, xLAM, Qwen3) get the tools natively; others get a prompt-based format.
  `tool_calls.py` parses `<tool_call>` blocks plus the near-misses small models produce (fenced
  JSON, Python-style calls) and records which form was used.
- **Grading** (`agent_bench.py`) reimplements BFCL's multi-turn checker without `eval()`: after every
  turn the agent's environment state must equal the state the reference calls produce, and the
  reference results must appear among the agent's results. Accuracy = every turn correct; "turns
  done" = turns passed before the first failure. Replaying the reference passes 200/200.
- **Trajectory analysis** (ideas from [TraceHarness](https://github.com/aura-lab-wm/traceharness), MIT):
  - an action string per run: **R** read · **W** write (changed the state) · **E** tool error ·
    **U** unknown tool · **C** call checked without execution (single-turn) · **A** text answer ·
    **X** unparseable output
  - step flags: repeat loop (same call 3× in a turn), error persists, unknown tool, format error,
    answered without acting, step limit
  - one failure mode per run, decided at the first failing turn
  - the first divergence between A and B, and of each against the reference
- **Views:** summary cards, a timeline of model steps, and a turn-by-turn grid with Reference | A | B
  lanes (results, raw output, state diff of the failing turn). History groups batches into runs; a
  run report shows accuracy, turns done, behaviour and cost per task, A-vs-B flips, failure modes
  and flags (click to filter), and every task's trajectory. Trajectories download as JSON.
- **Models:** the agent page has its own list (Qwen2.5-Instruct 0.5–7B, Hammer2.1, xLAM-2, Qwen3,
  Qwen2.5-Coder). Code models are not tuned for tool calling and tend to repeat the same call; the
  page warns when one is selected. Tool prompts are long (5–10k tokens per step), so leave memory for
  the KV cache.
- **Hosted:** run the `agent_runs` part of `deploy/supabase.sql` once. Backends started before this
  page existed need to be restarted with the latest `connect.py`.

## Models in the dropdown

Qwen2.5-Coder 0.5B/1.5B/3B/7B-Instruct, DeepSeek-Coder 1.3B/6.7B-Instruct,
CodeLlama-7B-Instruct, Granite-3B-Code-Instruct, Phi-3.5-mini-instruct, StarCoder2 3B/7B (base
models, completion only). Pre-quantized AWQ/GPTQ repos are listed but stay disabled until their
backend is installed (`autoawq`, or `optimum` + `auto-gptq`/`gptqmodel`); restart the server
after installing. Gated models (Llama, CodeGemma) need `huggingface-cli login` and the
"Custom repo…" option.

Measured on an L40S with Qwen2.5-Coder-7B-Instruct: FP16 uses 15.2 GB at ~42 tok/s, and 4-bit
NF4 uses 5.4 GB at ~29 tok/s.

## Precision options

| Option | What it does |
|---|---|
| `fp16` / `bf16` / `fp32` | Unquantized baseline |
| `bnb-8bit` | bitsandbytes LLM.int8(), quantized on the fly from the same repo (NVIDIA GPU only) |
| `bnb-4bit` | bitsandbytes NF4 + double quant, quantized on the fly (NVIDIA GPU only) |
| `quanto-int8` / `quanto-int4` | optimum-quanto weight-only quantization on the fly; also runs on CPU and Apple GPUs |
| `prequantized` | Loads an already-quantized repo (GPTQ / AWQ) |

## Notes on the numbers

- Use **Sequential** run mode for speed numbers. Parallel mode runs both models in one Python
  process, so they compete for the CPU and both report lower tok/s.
- bitsandbytes saves memory but usually decodes *slower* than FP16 at batch size 1.
- With temperature 0 (greedy decoding) runs are deterministic, so differences between A and B
  come from quantization.
- Benchmark grading mirrors each benchmark's harness (HumanEval: prompt + completion +
  `check()`; BigCodeBench: code + `unittest` in a registered module), but runs locally, not in
  the official sandbox. Sanity check on this machine: all 164 HumanEval reference solutions pass,
  and 144/148 BigCodeBench-Hard ones pass (the other 4 need network access, an audio library
  that fails here, or exceed the 60 s timeout). Absolute BigCodeBench scores can therefore be
  slightly lower than the leaderboard; the A-vs-B comparison is unaffected.
- **Generated code runs on this machine** in a subprocess (temporary working directory and
  HOME, no GPUs, timeout of `--test-timeout` / `--bench-timeout` seconds). It is not a security
  sandbox. BigCodeBench tasks do file and process work by design.
- Static analysis only parses the code; it never executes it.

## Files

- `server.py`: FastAPI backend (model slots, background loading, SSE streaming, grading and analysis endpoints)
- `runstore.py`: SQLite run history, per-run aggregation and issue tables
- `catalog.py`: model dropdown list
- `benchmarks.py`: HumanEval / BigCodeBench loading, prompts, code extraction and grading payloads
- `analysis.py`: ruff, pylint, radon, bandit
- `test_runner.py`: executes code + tests in an isolated subprocess
- `presets.py`: built-in tasks
- `static/index.html`: the single-page UI (local and hosted mode)
- `agent_bench.py`, `agent_benchmarks.py`, `agent_runner.py`, `tool_calls.py`, `static/agent.html`, `bfcl/`, `toolhop/`: the agent page
- `connect.py`: starts a backend (plus a tunnel unless `--local`) on Colab, a server or a laptop and prints a link to the hosted site
- `deploy/supabase.sql`, `deploy/build_site.py`: hosted setup and static site build
