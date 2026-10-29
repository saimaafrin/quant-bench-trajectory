# QuantBench in your codespace

You are reading this inside a GitHub Codespace that is setting itself up (or already did). Here is what
happens and what to do.

## First start (about four minutes, once)

The terminal at the bottom runs `.devcontainer/setup.sh`: it installs PyTorch (CPU build), the packages in
`requirements.txt`, and downloads the two default models, `Qwen/Qwen2.5-Coder-0.5B-Instruct` (code
generation page) and `Qwen/Qwen2.5-0.5B-Instruct` (agent page), about 1 GB each. When it prints
`QuantBench setup done.`, the server starts and a browser tab opens on **port 8000**.

If no tab opened: **Ports** tab (next to Terminal) → the row **QuantBench (8000)** → the globe icon
("Open in Browser"). The address looks like `https://<your-codespace>-8000.app.github.dev`. Only you can
open it; GitHub checks your login before any request reaches the server.

## Every later start

The codespace stops after 30 minutes idle (your work is kept: the run history is in `runs/history.sqlite`).
Reopen it from <https://github.com/codespaces>; the server starts by itself in a few seconds
(`.devcontainer/start.sh`). If the page says it cannot connect, wait ten seconds and reload.

## What to do on the page

1. **Load models**: click **Load model** on both cards. Defaults: A = FP32 (unquantized), B = INT8 weights
   (optimum-quanto). The models are already on disk, so each load takes 10-30 s; watch the **Load log**
   (tokenizer, weights, sanity check, warm-up). The card shows the weights' footprint: about 1.98 GB
   against 0.90 GB. INT8 is not a quarter of FP32 here because quanto converts only the linear layers;
   the embedding table (136M of the 494M parameters, shared with the output layer) stays in FP32.
2. **Coding task**: start with a built-in task (they have unit tests), then try **HumanEval** (164
   problems; downloads from the Hub on first use). **Generate ▶** streams both answers; under each you get
   the tests and the static analysis; the **Comparison** block gives memory saving, speed ratio, pass/fail
   and the output similarity between A and B.
3. **Batch evaluation** (with a benchmark source): N problems in a row and a report with pass@1 per
   model, every problem's outcome (both pass / only A / only B / both fail), and CSV export.
4. **Tool-calling agents** (the second page, top left): the same two slots run as agents on BFCL
   multi-turn tasks; the trajectories of A, B and the reference are shown side by side, with the first
   step where they diverge. On CPU, start with **BFCL single-turn** (a few seconds per task; the 0.5B models
   pass most of them), then move to multi-turn (about a minute per task): the 0.5B models fail most of
   those, and A and B often fail the same way, so look for tasks where they part (in our run,
   `multi_turn_base_2` split at its first call: `mkdir('TeamNotes')` against `mkdir('team_notes')`).

On a codespace (4 cores, no GPU) expect **3-8 tokens per second** per model, so keep *Max new tokens* at
256-512 and run batches of 10-20 problems, not the whole benchmark. The 1.5B models fit too (about 6 GB
in FP32) but are slow; anything larger needs a GPU. Slot B is the slower one: on a CPU, quanto's INT8
weights save memory, not time, and B decodes about 2.5 times slower than A (60 against 24 tokens per
second on four Apple M4 cores). That is a result to discuss, not a fault.

## If something goes wrong

- The server's log: `cat /tmp/quantbench.log` in the terminal. Restart it with `bash .devcontainer/start.sh`.
- A model card says *Loaded, but this model's outputs look broken*: the sanity check (perplexity of a plain
  English sentence) failed; unload and load again, or pick another precision. Report it with the load log.
- Out of memory (the page stops answering): the codespace machine has 16 GB; unload one model, or use the
  0.5B models only. `free -h` shows what is left.
- Free GitHub accounts get 120 core-hours of Codespaces per month (a 4-core machine uses 4 per hour);
  students on the GitHub Student Developer Pack get 180. Stop the codespace when you are done
  (<https://github.com/codespaces> → ... → Stop).
