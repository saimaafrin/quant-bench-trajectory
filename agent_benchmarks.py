"""The agent page's tool-calling benchmarks, behind one interface (Benchmark).

- bfcl_multi:  BFCL v4 multi-turn base, 200 stateful multi-turn tasks (agent_bench.py)
- bfcl_single: BFCL v4 single-turn AST categories, 1,240 tasks (bfcl/NOTICE.md)
- toolhop:     ToolHop multi-hop tool use, 995 tasks with executable tools (toolhop/NOTICE.md)
"""
import ast
import copy
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import agent_bench as mt   # BFCL multi-turn

ROOT = Path(__file__).resolve().parent


class Benchmark:
    """What agent_runner needs from a benchmark.

    key / name / about / filter_label: shown in the UI
    max_steps: default steps per user turn; executes: whether tool calls are run at all
    ref_kind: "calls" (reference calls comparable to the agent's), "accepted" (calls with every
              accepted argument value), or "subanswers" (intermediate answers, not calls)
    """
    key = name = about = filter_label = ""
    max_steps, executes, ref_kind = 20, True, "calls"

    def listing(self) -> list[dict]:
        raise NotImplementedError

    def task(self, index: int) -> dict:
        raise NotImplementedError

    def detail(self, index: int) -> dict:
        raise NotImplementedError

    def tools(self, task: dict) -> list[dict]:
        raise NotImplementedError

    def user_turns(self, task: dict) -> list[str]:
        return task["turns"]

    def system(self, task: dict, native: bool):
        return None

    def env(self, task: dict):
        raise NotImplementedError

    def grade(self, task: dict, turns: list, steps: list) -> dict:
        raise NotImplementedError

    def ref_calls(self, task: dict) -> int:
        return 0

    def info(self) -> dict:
        return {"key": self.key, "name": self.name, "about": self.about, "filter_label": self.filter_label,
                "max_steps": self.max_steps, "executes": self.executes, "ref_kind": self.ref_kind}


MULTI_TURN_BEHAVIOR = (   # BFCL's own multi-turn instruction (bfcl_eval/constants/default_prompts.py)
    "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
    "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
    "Once you have no more functions to call, the system will consider the current turn complete and proceed to "
    "the next turn or task.")


# ---------------------------------------------------------------- BFCL multi-turn
class BfclMulti(Benchmark):
    key, name, filter_label = "bfcl_multi", "BFCL multi-turn", "Environment"
    about = ("Berkeley Function Calling Leaderboard v4, multi-turn base: 200 conversations of 1–7 user turns against "
             "stateful, executable tools. A turn passes when the environment ends in the reference state.")
    max_steps = mt.MAX_STEPS

    def listing(self):
        return [{**x, "groups": x["envs"]} for x in mt.listing()]

    def task(self, index):
        return mt.tasks()[index]

    def detail(self, index):
        d = mt.task_detail(index)
        return {**d, "groups": d["envs"]}

    def tools(self, task):
        return mt.tool_schemas(task["classes"])

    def system(self, task, native):
        return ("You are a helpful assistant that completes the user's requests by calling the available tools. "
                + MULTI_TURN_BEHAVIOR)

    def env(self, task):
        return mt.Environment(task)

    def grade(self, task, turns, steps):
        return mt.grade(task, turns)

    def ref_calls(self, task):
        return sum(map(len, task["ground_truth"]))


# ---------------------------------------------------------------- BFCL single-turn (AST)
SINGLE = ROOT / "bfcl" / "data" / "single"
SINGLE_CATEGORIES = {"simple_python": "Simple", "multiple": "Multiple", "parallel": "Parallel",
                     "parallel_multiple": "Parallel multiple", "irrelevance": "Irrelevance"}


class NoExecution:
    """Single-turn BFCL checks the call itself; nothing is executed."""

    def __init__(self, functions):
        self.names = {f["name"] for f in functions}

    def call(self, name, args):
        return "(not executed: this benchmark checks the call itself)"

    def owner(self, name):
        return "function" if name in self.names else None

    def snapshot(self):
        return {}

    def close(self):
        pass


@lru_cache(maxsize=1)
def _single_tasks():
    out = []
    for cat, label in SINGLE_CATEGORIES.items():
        qs = [json.loads(line) for line in (SINGLE / f"{cat}.jsonl").read_text().splitlines() if line.strip()]
        ans_path = SINGLE / f"{cat}_answers.jsonl"
        answers = ({a["id"]: a["ground_truth"] for a in map(json.loads, ans_path.read_text().splitlines()) if a}
                   if ans_path.exists() else {})
        for q in qs:
            turns = [" ".join(m["content"] for m in turn if m["role"] == "user") for turn in q["question"]]
            out.append({"id": q["id"], "category": cat, "label": label, "turns": turns[:1],
                        "functions": q["function"], "ground_truth": answers.get(q["id"], [])})
    return out


class BfclSingle(Benchmark):
    key, name, filter_label = "bfcl_single", "BFCL single-turn", "Category"
    about = ("Berkeley Function Calling Leaderboard v4, single-turn: pick the right function(s) and arguments in one "
             "reply (simple, multiple, parallel, parallel multiple), or call nothing when no tool fits (irrelevance). "
             "Graded with BFCL's AST checker.")
    max_steps, executes, ref_kind = 1, False, "accepted"

    def listing(self):
        return [{"index": i, "id": t["id"], "turns": 1, "gt_calls": len(t["ground_truth"]), "groups": [t["label"]],
                 "title": t["turns"][0][:90]} for i, t in enumerate(_single_tasks())]

    def task(self, index):
        return _single_tasks()[index]

    def tools(self, task):
        return [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                  "parameters": mt._json_schema(f["parameters"])}} for f in task["functions"]]

    def system(self, task, native):
        return None if native else "You are an expert in composing functions."

    def detail(self, index):
        t = self.task(index)
        calls = []
        for g in t["ground_truth"]:
            for name, params in g.items():
                first = {k: v[0] for k, v in params.items() if v and v[0] != ""}
                accepted = "; ".join(f"{k}: {' | '.join(map(repr, v))}" for k, v in params.items())
                calls.append({"name": name, "arguments": first, "text": mt.call_text(name, first), "action": "C",
                              "result": f"Accepted values: {accepted}" if accepted else "No arguments"})
        note = "No function fits this request: the agent should not call any." if t["category"] == "irrelevance" else None
        return {"index": index, "id": t["id"], "turns": t["turns"], "groups": [t["label"]], "envs": [t["label"]],
                "tools": [f["name"] for f in t["functions"]], "classes": [],
                "reference": [{"turn": 0, "user": t["turns"][0], "calls": calls, "note": note}]}

    def env(self, task):
        return NoExecution(task["functions"])

    def grade(self, task, turns, steps):
        from bfcl.ast_checker import Language, ast_checker
        calls = [c for s in steps if s["turn"] == 0 for c in s["calls"]]
        output = [{c["name"]: c["arguments"] if isinstance(c["arguments"], dict) else {}} for c in calls]
        if task["category"] == "irrelevance":
            ok = not calls
            check = {"ok": ok, "error_type": None if ok else "called_irrelevant",
                     "detail": None if ok else {"errors": ["Called a function although none fits the request"]}}
        elif not calls:
            check = {"ok": False, "error_type": "no_call", "detail": {"errors": ["No function call in the reply"]}}
        else:
            try:
                r = ast_checker(task["functions"], output, task["ground_truth"], Language.PYTHON, task["category"], "local")
            except Exception as e:  # noqa: BLE001 - model output the checker cannot handle
                r = {"valid": False, "error": [f"{type(e).__name__}: {e}"], "error_type": "checker_error"}
            check = {"ok": r["valid"], "error_type": None if r["valid"] else r.get("error_type", "ast_mismatch"),
                     "detail": None if r["valid"] else {"errors": [str(x) for x in r.get("error", [])][:6]}}
        check.update(turn=0, after_failure=False, gt_calls=[])
        return {"valid": check["ok"], "turns": [check], "turns_ok": int(check["ok"]), "turns_total": 1,
                "first_error": None if check["ok"] else {"turn": 0, "error_type": check["error_type"]}}

    def ref_calls(self, task):
        return len(task["ground_truth"])


# ---------------------------------------------------------------- ToolHop (multi-hop, executable tools)
TOOLHOP_PROMPT = (   # "Mandatory" scenario of ToolHop's code/evaluation_open.py, verbatim (typo included)
    "You will be asked a question with some tools, and should provide a short final answer.\n"
    "Please note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\n"
    "If the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\n"
    "If the final nswer is a name, format it as follows: Firstname Lastname\n"
    "If the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\n"
    "Please provide the final answer in the following format: <answer>final answer here</answer>\n"
    "Answer as short as possible.\nQuestion: {question}")
TOOL_TIMEOUT = 10   # seconds per call, as in ToolHop's evaluation
WORKER = ROOT / "toolhop" / "worker.py"


class ToolSandbox:
    """ToolHop's tools are Python source from the dataset: they run in a separate `python -I`
    process with an empty working directory, no GPUs and a per-call time limit, never in the
    server process."""

    def __init__(self, task):
        self.functions = {f.split("def", 1)[1].split("(", 1)[0].strip(): f for f in task["functions"]}
        self.dir = tempfile.TemporaryDirectory(prefix="toolhop-")
        self.proc = None

    def _start(self):
        env = {**os.environ, "HOME": self.dir.name, "TMPDIR": self.dir.name, "CUDA_VISIBLE_DEVICES": "",
               "PYTHONIOENCODING": "utf-8"}
        self.proc = subprocess.Popen([sys.executable, "-I", str(WORKER)], cwd=self.dir.name, env=env, text=True,
                                     encoding="utf-8", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        self.replies = queue.Queue()
        threading.Thread(target=lambda out=self.proc.stdout, q=self.replies: [q.put(line) for line in out], daemon=True).start()

    def call(self, name, args):
        if name not in self.functions:
            return f"an error occured when call {name}: no such tool"
        if not isinstance(args, dict):
            return f"an error occured when call {name}: arguments must be a JSON object"
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self.proc.stdin.write(json.dumps({"source": self.functions[name], "name": name, "args": args}, default=str) + "\n")
            self.proc.stdin.flush()
            reply = json.loads(self.replies.get(timeout=TOOL_TIMEOUT))
            if not isinstance(reply, dict) or "ok" not in reply:
                raise ValueError("unexpected reply from the tool process")
        except queue.Empty:
            self.proc.kill()
            return f"an error occured when call {name}: timed out after {TOOL_TIMEOUT}s"
        except (OSError, ValueError) as e:
            self.proc.kill()
            return f"an error occured when call {name}: {e}"
        if not reply["ok"]:
            return f"an error occured when call {name}: {reply['error']}"
        return json.dumps(reply["result"], ensure_ascii=False)

    def owner(self, name):
        return "tool" if name in self.functions else None

    def snapshot(self):
        return {}

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.dir.cleanup()


@lru_cache(maxsize=1)
def _toolhop_tasks():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("bytedance-research/ToolHop", "data/ToolHop.json", repo_type="dataset")
    data = json.loads(Path(path).read_text())
    return [{"id": f"toolhop_{d['id']}", "question": d["question"], "answer": str(d["answer"]), "sub_task": d["sub_task"],
             "tools": list(d["tools"].values()), "functions": d["functions"], "domain": d.get("domain"),
             "answer_type": d.get("answer_type"), "turns": [d["question"]]} for d in data]


class ToolHop(Benchmark):
    key, name, filter_label = "toolhop", "ToolHop", "Hops"
    about = ("ToolHop (ACL 2025): 995 multi-hop questions, each answered by chaining 3–7 executable tools. "
             "Graded on the final answer as in ToolHop's official evaluation (tools are mandatory, at most 9 model turns).")
    max_steps, ref_kind = 9, "subanswers"

    def listing(self):
        return [{"index": i, "id": t["id"], "turns": 1, "gt_calls": len(t["sub_task"]), "groups": [f"{len(t['sub_task'])} hops"],
                 "title": t["question"][:90]} for i, t in enumerate(_toolhop_tasks())]

    def task(self, index):
        return _toolhop_tasks()[index]

    def tools(self, task):
        return [{"type": "function", "function": copy.deepcopy(t)} for t in task["tools"]]

    def user_turns(self, task):
        return [TOOLHOP_PROMPT.format(question=task["question"])]

    def detail(self, index):
        t = self.task(index)
        calls = [{"name": "", "arguments": {}, "text": q, "result": a, "action": "R"} for q, a in t["sub_task"].items()]
        return {"index": index, "id": t["id"], "turns": t["turns"], "groups": [f"{len(t['sub_task'])} hops"],
                "envs": [t["domain"] or "ToolHop"], "tools": [x["name"] for x in t["tools"]], "classes": [], "answer": t["answer"],
                "reference": [{"turn": 0, "user": t["question"], "calls": calls, "note": f"Expected final answer: {t['answer']}"}]}

    def env(self, task):
        return ToolSandbox(task)

    def grade(self, task, turns, steps):
        final = steps[-1] if steps and not steps[-1]["calls"] else None
        answer = final["content"] or final["raw"] if final else None
        last_tool = None
        if final and len(steps) > 1 and steps[-2]["calls"]:
            last_tool = steps[-2]["calls"][-1]["result"]
        ok, got = toolhop_correct(task["answer"], answer, last_tool)
        check = {"turn": 0, "after_failure": False, "gt_calls": [], "ok": ok,
                 "error_type": None if ok else ("no_answer" if answer is None else "wrong_answer"),
                 "detail": None if ok else {"expected": task["answer"], "got": got}}
        return {"valid": ok, "turns": [check], "turns_ok": int(ok), "turns_total": 1, "answer": got, "expected": task["answer"],
                "first_error": None if ok else {"turn": 0, "error_type": check["error_type"]}}

    def ref_calls(self, task):
        return len(task["sub_task"])


def toolhop_correct(expected: str, solution, last_tool_output):
    """ToolHop's answer check (evaluation_open.py), with ast.literal_eval in place of eval."""
    if solution is None:
        return False, None
    got = solution.split("<answer>")[-1].split("</answer>")[0] if "<answer>" in solution else solution

    def norm(x):
        return str(x).removesuffix(".0").lower()

    ok = False
    try:
        truth = ast.literal_eval(expected.strip())
    except (ValueError, SyntaxError):
        ok = norm(expected) in norm(got).replace(",", "")
    else:
        try:
            ok = ast.literal_eval(got.strip()) == truth
        except (ValueError, SyntaxError):
            ok = False
    if last_tool_output is not None and norm(expected) in norm(last_tool_output).replace(",", ""):
        ok = True
    return ok, got.strip()


BENCHMARKS = {b.key: b for b in (BfclMulti(), BfclSingle(), ToolHop())}
