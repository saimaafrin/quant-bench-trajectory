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
   (tokenizer, weights, sanity check, warm-up). The card shows the weights' footprint: about 1.96 GB
   against 0.56 GB.
2. **Coding task**: start with a built-in task (they have unit tests), then try **HumanEval** (164
   problems; downloads from the Hub on first use). **Generate ▶** streams both answers; under each you get
   the tests and the static analysis; the **Comparison** block gives memory saving, speed ratio, pass/fail
   and the output similarity between A and B.
3. **Batch evaluation** (with a benchmark source): N problems in a row and a report with pass@1 per
   model, every problem's outcome (both pass / only A / only B / both fail), and CSV export.
4. **Tool-calling agents** (the second page, top left): the same two slots run as agents on BFCL
   multi-turn tasks; the trajectories of A, B and the reference are shown side by side, with the first
   step where they diverge.

On a codespace (4 cores, no GPU) expect **3-8 tokens per second** per model, so keep *Max new tokens* at
256-512 and run batches of 10-20 problems, not the whole benchmark. The 1.5B models fit too (about 6 GB
in FP32) but are slow; anything larger needs a GPU.

## If something goes wrong

- The server's log: `cat /tmp/quantbench.log` in the terminal. Restart it with `bash .devcontainer/start.sh`.
- A model card says *Loaded, but this model's outputs look broken*: the sanity check (perplexity of a plain
  English sentence) failed; unload and load again, or pick another precision. Report it with the load log.
- Out of memory (the page stops answering): the codespace machine has 16 GB; unload one model, or use the
  0.5B models only. `free -h` shows what is left.
- Free GitHub accounts get 120 core-hours of Codespaces per month (a 4-core machine uses 4 per hour);
  students on the GitHub Student Developer Pack get 180. Stop the codespace when you are done
  (<https://github.com/codespaces> → ... → Stop).
