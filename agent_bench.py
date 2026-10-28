"""Tool-calling benchmark for the agent page: BFCL v4 multi-turn (base), 200 tasks.

Each task is a short conversation (1-7 user turns) against stateful, executable tool
environments (file system, trading bot, travel booking, vehicle control, messaging, ...).
Within a turn the agent may call tools for up to MAX_STEPS steps; the turn ends when it
answers without a tool call.

Grading follows BFCL's multi-turn checker (bfcl/NOTICE.md): after every turn the model's
environment state must equal the state reached by replaying the ground-truth calls, and the
results of the ground-truth calls of that turn must all appear among the results the model
has produced so far. Nothing is eval()'d: model calls arrive as JSON and are dispatched to
public methods; ground-truth calls are parsed with ast and must contain only literals.
"""
import ast
import copy
import importlib
import inspect
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "bfcl" / "data"
MAX_STEPS = 20   # per user turn, as in BFCL (MAXIMUM_STEP_LIMIT)

CLASSES = {   # class name -> (module, func-doc file)
    "GorillaFileSystem": ("gorilla_file_system", "gorilla_file_system"),
    "MathAPI": ("math_api", "math_api"),
    "MessageAPI": ("message_api", "message_api"),
    "TwitterAPI": ("posting_api", "posting_api"),
    "TicketAPI": ("ticket_api", "ticket_api"),
    "TradingBot": ("trading_bot", "trading_bot"),
    "TravelAPI": ("travel_booking", "travel_booking"),
    "VehicleControlAPI": ("vehicle_control", "vehicle_control"),
}
STATELESS = {"MathAPI"}
LABELS = {"GorillaFileSystem": "File system", "MathAPI": "Math", "MessageAPI": "Messaging", "TwitterAPI": "Posting",
          "TicketAPI": "Tickets", "TradingBot": "Trading", "TravelAPI": "Travel", "VehicleControlAPI": "Vehicle"}


@lru_cache(maxsize=1)
def tasks() -> list[dict]:
    answers = {a["id"]: a["ground_truth"] for a in map(json.loads, (DATA / "answers.jsonl").read_text().splitlines())}
    out = []
    for q in map(json.loads, (DATA / "questions.jsonl").read_text().splitlines()):
        turns = [" ".join(m["content"] for m in turn if m["role"] == "user") for turn in q["question"]]
        out.append({"id": q["id"], "turns": turns, "initial_config": q["initial_config"],
                    "classes": q["involved_classes"], "ground_truth": answers[q["id"]]})
    return out


def listing() -> list[dict]:
    return [{"index": i, "id": t["id"], "turns": len(t["turns"]), "gt_calls": sum(map(len, t["ground_truth"])),
             "envs": [LABELS[c] for c in t["classes"]], "title": t["turns"][0][:90]} for i, t in enumerate(tasks())]


@lru_cache(maxsize=None)
def func_docs(class_name: str) -> list[dict]:
    path = DATA / "func_doc" / f"{CLASSES[class_name][1]}.json"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tool_schemas(classes: list[str]) -> list[dict]:
    """OpenAI-style tool definitions (what chat templates expect)."""
    tools = []
    for c in classes:
        for d in func_docs(c):
            params = _json_schema(d["parameters"])
            tools.append({"type": "function", "function": {"name": d["name"], "description": d["description"], "parameters": params}})
    return tools


TYPE_NAMES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _json_schema(p):
    # BFCL uses Python-flavoured type names ("dict", "float", "tuple", "any"). Keys under
    # "properties" are parameter names (one may be called "type"), so walk them separately.
    if isinstance(p, list):
        return [_json_schema(x) for x in p]
    if not isinstance(p, dict):
        return p
    out = {}
    for k, v in p.items():
        if k == "response":
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: _json_schema(sub) for name, sub in v.items()}
        elif k == "type" and isinstance(v, str):
            out[k] = TYPE_NAMES.get(v, v)
        else:
            out[k] = _json_schema(v)
    return out


class Environment:
    """Live instances of a task's tool classes. Never touches the real file system or network."""

    def __init__(self, task: dict):
        self.instances, self.methods = {}, {}
        for c in task["classes"]:
            module = importlib.import_module(f"bfcl.envs.{CLASSES[c][0]}")
            inst = getattr(module, c)()
            if c not in STATELESS:
                inst._load_scenario(copy.deepcopy(task["initial_config"].get(c, {})), long_context=False)
            self.instances[c] = inst
            for name, method in inspect.getmembers(inst, predicate=inspect.ismethod):
                if not name.startswith("_"):
                    self.methods[name] = (c, method)

    def call(self, name: str, args) -> str:
        if name not in self.methods:
            return f"Error during execution: unknown function '{name}'"
        try:
            if isinstance(args, dict):
                result = self.methods[name][1](**args)
            elif isinstance(args, (list, tuple)):
                result = self.methods[name][1](*args)
            else:
                raise TypeError("arguments must be a JSON object")
        except Exception as e:  # noqa: BLE001 - errors are part of the trajectory, as in BFCL
            return f"Error during execution: {e}"
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            try:
                return json.dumps(result)
            except TypeError:
                return str(result)
        return str(result)

    def owner(self, name: str):
        return self.methods[name][0] if name in self.methods else None

    def snapshot(self) -> dict:
        """Public attributes of every instance (deep-copied), for the per-turn state check."""
        return {c: {k: copy.deepcopy(v) for k, v in vars(i).items() if not k.startswith("_")} for c, i in self.instances.items()}


def parse_call(text: str):
    """"mv(source='a', destination='b')" -> ("mv", args, kwargs), literals only."""
    node = ast.parse(text.strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"not a simple call: {text}")
    return node.func.id, [ast.literal_eval(a) for a in node.args], {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def call_text(name: str, args) -> str:
    if isinstance(args, dict):
        return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
    return f"{name}({', '.join(map(repr, args or []))})"


def ground_truth_run(task: dict) -> list[dict]:
    """Replay the reference calls turn by turn: results and the state after each turn."""
    env, out = Environment(task), []
    for turn_calls in task["ground_truth"]:
        results = []
        for text in turn_calls:
            name, args, kwargs = parse_call(text)
            if args and kwargs:   # positional + keyword: map positionals to parameter names
                params = list(inspect.signature(env.methods[name][1]).parameters)
                kwargs = {**dict(zip(params, args)), **kwargs}
                args = []
            results.append({"call": text, "result": env.call(name, kwargs if kwargs or not args else args)})
        out.append({"calls": results, "state": env.snapshot()})
    return out


def grade(task: dict, model_turns: list[dict]) -> dict:
    """model_turns[i] = {"results": [tool results of turn i], "state": Environment.snapshot() after turn i}.

    Returns BFCL's verdict for the whole task plus a per-turn breakdown (every turn is checked,
    so the page can show where a trajectory first went wrong)."""
    gt = ground_truth_run(task)
    seen_results, per_turn, first_error = [], [], None
    for i, ref in enumerate(gt):
        mine = model_turns[i] if i < len(model_turns) else {"results": [], "state": None, "missing": True}
        seen_results += mine["results"]
        check = {"turn": i, "gt_calls": [c["call"] for c in ref["calls"]], "ok": True, "error_type": None, "detail": None}
        if ref["calls"] and not mine["results"]:
            check.update(ok=False, error_type="empty_turn", detail="No tool was called in this turn")
        elif ref["calls"]:
            diffs = state_diff(mine["state"] or {}, ref["state"])
            missing = _missing(ref_results := [c["result"] for c in ref["calls"]], seen_results)
            if diffs:
                check.update(ok=False, error_type="state_mismatch", detail=diffs)
            elif missing:
                check.update(ok=False, error_type="response_mismatch",
                             detail={"missing_results": missing, "expected": ref_results})
        # Ground truth assumes every earlier turn went right, so after the first failure the
        # checks still run (to show what differs) but no longer count as progress.
        check["after_failure"] = first_error is not None
        per_turn.append(check)
        if not check["ok"] and first_error is None:
            first_error = check
    progress = first_error["turn"] if first_error else len(per_turn)
    return {"valid": first_error is None, "turns": per_turn,
            "first_error": first_error and {"turn": first_error["turn"], "error_type": first_error["error_type"]},
            "turns_ok": progress, "turns_total": len(per_turn)}


def state_diff(model_state: dict, gt_state: dict) -> dict:
    diffs = {}
    for c, attrs in gt_state.items():
        if c in STATELESS:
            continue
        for k, v in attrs.items():
            mv = (model_state.get(c) or {}).get(k)
            if mv != v:
                diffs[f"{c}.{k}"] = {"model": to_jsonable(mv), "ground_truth": to_jsonable(v)}
    return diffs


def _missing(expected: list, have: list) -> list:
    pool, missing = list(have), []
    for item in expected:
        try:
            pool.remove(item)
        except ValueError:
            missing.append(item)
    return missing


def to_jsonable(v, depth=0):
    if depth > 8:
        return "…"
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): to_jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [to_jsonable(x, depth + 1) for x in v]
    if hasattr(v, "__dict__"):   # e.g. File / Directory objects of the file system
        return {k: to_jsonable(x, depth + 1) for k, x in vars(v).items() if not k.startswith("_") and k != "parent"}
    return str(v)


def reference(task: dict) -> list[dict]:
    """The ground-truth trajectory as the page draws it: per turn, each call with its result and
    action letter (W if it changed the environment, R otherwise)."""
    env, out = Environment(task), []
    for ti, turn_calls in enumerate(task["ground_truth"]):
        calls = []
        for text in turn_calls:
            name, args, kwargs = parse_call(text)
            if args:
                params = list(inspect.signature(env.methods[name][1]).parameters)
                kwargs, args = {**dict(zip(params, args)), **kwargs}, []
            before = env.snapshot()
            result = env.call(name, kwargs)
            calls.append({"name": name, "arguments": kwargs, "text": call_text(name, kwargs), "env": env.owner(name),
                          "result": result, "action": "W" if env.snapshot() != before else "R"})
        out.append({"turn": ti, "user": task["turns"][ti], "calls": calls})
    return out


def task_detail(index: int) -> dict:
    t = tasks()[index]
    return {"index": index, "id": t["id"], "turns": t["turns"], "classes": t["classes"],
            "envs": [LABELS[c] for c in t["classes"]], "tools": [x["function"]["name"] for x in tool_schemas(t["classes"])],
            "reference": reference(t), "initial_state": to_jsonable(Environment(t).snapshot())}
