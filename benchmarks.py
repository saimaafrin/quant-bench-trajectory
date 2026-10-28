"""HumanEval and BigCodeBench problems: loading, prompting and building the code to grade."""
import re
import threading

import pandas as pd
from huggingface_hub import hf_hub_download

# Same instruction BigCodeBench uses for its "instruct" split; reused for HumanEval for consistency.
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

BENCHMARKS = {
    "humaneval": {"name": "HumanEval", "repo": "openai/openai_humaneval",
                  "file": "openai_humaneval/test-00000-of-00001.parquet", "count": 164,
                  "kind": "humaneval", "max_new_tokens": 512},
    "bigcodebench-hard": {"name": "BigCodeBench-Hard (instruct)", "repo": "bigcode/bigcodebench-hard",
                          "file": "data/v0.1.4-00000-of-00001.parquet", "count": 148,
                          "kind": "unittest", "max_new_tokens": 1024},
    "bigcodebench": {"name": "BigCodeBench full (instruct)", "repo": "bigcode/bigcodebench",
                     "file": "data/v0.1.4-00000-of-00001.parquet", "count": 1140,
                     "kind": "unittest", "max_new_tokens": 1024},
}

_cache: dict[str, list[dict]] = {}
_lock = threading.Lock()


def load(name: str) -> list[dict]:
    if name not in BENCHMARKS:
        raise KeyError(name)
    with _lock:
        if name not in _cache:
            meta = BENCHMARKS[name]
            df = pd.read_parquet(hf_hub_download(meta["repo"], meta["file"], repo_type="dataset"))
            _cache[name] = df.to_dict("records")
        return _cache[name]


def listing(name: str) -> list[dict]:
    rows = load(name)
    if BENCHMARKS[name]["kind"] == "humaneval":
        return [{"index": i, "task_id": r["task_id"], "title": r["entry_point"]} for i, r in enumerate(rows)]
    return [{"index": i, "task_id": r["task_id"], "title": _first_sentence(r["instruct_prompt"])} for i, r in enumerate(rows)]


def problem(name: str, index: int) -> dict:
    meta, r = BENCHMARKS[name], load(name)[index]
    if meta["kind"] == "humaneval":
        prompt = f"{INSTRUCTION}\n```python\n{r['prompt'].strip()}\n```"
    else:
        prompt = f"{INSTRUCTION}\n```\n{r['instruct_prompt'].strip()}\n```"
    return {"benchmark": name, "index": index, "task_id": r["task_id"], "entry_point": r["entry_point"],
            "prompt": prompt, "system": "", "test": r["test"], "kind": meta["kind"],
            "max_new_tokens": meta["max_new_tokens"]}


def build_grading(name: str, index: int, completion: str) -> dict:
    """Return the runner payload {code, tests, kind} for a model completion."""
    meta, r = BENCHMARKS[name], load(name)[index]
    entry = r["entry_point"]
    code = extract_code(completion, entry)
    defines_entry = re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, flags=re.M) is not None
    if meta["kind"] == "humaneval":
        # Keep imports and helper functions that the prompt defines before the entry point.
        prefix = r["prompt"][: r["prompt"].find(f"def {entry}")]
        if not defines_entry:
            full = r["prompt"] + code
            return {"code": full, "analysis_code": full, "tests": [r["test"] + f"\ncheck({entry})"], "kind": "asserts"}
        # Static analysis sees the model's code plus only the prompt lines it didn't already write,
        # so re-imports of the prompt's own imports aren't reported as the model's issues.
        own = {l.strip() for l in code.splitlines()}
        needed = "\n".join(l for l in prefix.splitlines() if l.strip() not in own)
        return {"code": prefix + "\n" + code, "analysis_code": needed + "\n" + code,
                "tests": [r["test"] + f"\ncheck({entry})"], "kind": "asserts"}
    if not defines_entry:
        code = r["code_prompt"] + code
    return {"code": code, "tests": [r["test"]], "kind": "unittest"}


def extract_code(text: str, entry_point: str | None = None) -> str:
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, flags=re.S)
    if not blocks:  # unterminated block (hit max tokens) or no fences at all
        m = re.search(r"```(?:python|py|Python)?[ \t]*\n(.*)", text, flags=re.S)
        blocks = [m.group(1)] if m else [text]
    # Models sometimes put the language tag on its own line ("```\npython\n...").
    blocks = [re.sub(r"\A\s*(?:python|py|Python)[ \t]*\n", "", b) for b in blocks]
    if entry_point:
        for b in blocks:
            if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b):
                return b
    return "\n\n".join(blocks)


def _first_sentence(text: str, limit: int = 90) -> str:
    s = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return s if len(s) <= limit else s[: limit - 1] + "…"
