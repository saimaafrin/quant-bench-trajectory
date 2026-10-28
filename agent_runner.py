"""Run one local HF model as a tool-calling agent on one benchmark task, step by step.

Every step is recorded (prompt size, raw model output, parsed calls, tool results, timing),
which is what the agent page draws as a trajectory. Events are pushed to `emit` while the
episode runs so the page can animate both agents live.
"""
import json
import threading
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import agent_bench
import tool_calls

SYSTEM = ("You are a helpful assistant that completes the user's requests by calling the available tools. "
          "Keep calling tools until the request is fully handled, then reply briefly in plain text.")
LOOP_GUARD = 3   # end a turn once this many of its steps have made the identical call (0 = never)


class _Stop(StoppingCriteria):
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, input_ids, scores, **kw):
        return torch.full((input_ids.shape[0],), self.flag.is_set(), dtype=torch.bool, device=input_ids.device)


class _Streamer(TextIteratorStreamer):
    def __init__(self, tok):
        super().__init__(tok, skip_prompt=True, skip_special_tokens=True)
        self.n_tokens, self.first = 0, None

    def put(self, value):
        if not (self.skip_prompt and self.next_tokens_are_prompt):
            self.first = self.first or time.perf_counter()
            self.n_tokens += value.numel()
        super().put(value)


def native_tools(tok) -> bool:
    """Does the chat template render a tool list (and so know a tool-call format)?"""
    if not tok.chat_template:
        return False
    probe = [{"type": "function", "function": {"name": "zz_probe_tool", "description": "", "parameters": {"type": "object", "properties": {}}}}]
    try:
        return "zz_probe_tool" in tok.apply_chat_template([{"role": "user", "content": "hi"}], tools=probe,
                                                          tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 - templates that reject the argument
        return False


def renders_tool_calls(tok) -> bool:
    """Does the chat template show an assistant message's tool_calls? (Some, e.g. Hammer2.1's, drop them.)"""
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "zz_probe_call", "arguments": {}}}]},
            {"role": "tool", "name": "zz_probe_call", "content": "ok"}]
    try:
        return "zz_probe_call" in tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        return False


def run_episode(slot, bench, task: dict, opts: dict, emit, stop_flag, no_grad) -> dict:
    """Drive one agent through one task of `bench` (agent_benchmarks.Benchmark).

    slot has .model .tok .device .precision; returns the trajectory record for this agent."""
    tok, model, device = slot.tok, slot.model, slot.device
    tools = bench.tools(task)
    native = native_tools(tok)
    structured_history = native and renders_tool_calls(tok)
    max_steps = int(opts.get("max_steps") or bench.max_steps) if bench.executes else 1
    max_new = int(opts.get("max_new_tokens", 512))
    loop_guard = int(opts.get("loop_guard", LOOP_GUARD))   # 0 = off (pure benchmark behaviour)
    ctx_limit = getattr(model.config, "max_position_embeddings", None) or 32768

    system = bench.system(task, native)
    if not native:
        system = ((system or SYSTEM) + "\n\n" + tool_calls.FALLBACK_PROMPT.format(
            tools="\n".join(json.dumps(t["function"]) for t in tools)))
    messages = [{"role": "system", "content": system}] if system else []
    env = bench.env(task)
    steps, turns, t_start = [], [], time.perf_counter()
    stopped_reason = None
    try:
        for ti, user_text in enumerate(bench.user_turns(task)):
            messages.append({"role": "user", "content": user_text})
            emit({"type": "turn", "turn": ti, "text": user_text})
            turn_results, turn_end, seen = [], "answered", {}
            for si in range(max_steps):
                if stop_flag.is_set():
                    turn_end = stopped_reason = "stopped"
                    break
                prompt = _render(tok, messages, tools if native else None)
                inputs = tok(prompt, return_tensors="pt").to(device)
                n_prompt = inputs["input_ids"].shape[1]
                if n_prompt + max_new > ctx_limit:
                    turn_end = stopped_reason = "context_full"
                    break
                emit({"type": "step_start", "turn": ti, "step": si, "prompt_tokens": n_prompt})
                text, gen = _generate(model, tok, inputs, max_new, opts, stop_flag, no_grad,
                                      lambda chunk, n: emit({"type": "token", "turn": ti, "step": si, "text": chunk, "n": n}))
                parsed = tool_calls.parse(text)
                step = {"turn": ti, "step": si, "prompt_tokens": n_prompt, "raw": text, "format": parsed["format"],
                        "content": parsed["content"], "parse_errors": parsed["errors"], "calls": [], **gen,
                        "t": round(time.perf_counter() - t_start, 3)}
                if gen["hit_max_tokens"] and not parsed["calls"]:
                    step["parse_errors"] = step["parse_errors"] or ["output cut off at the token limit"]
                if not parsed["calls"]:
                    steps.append(step)
                    emit({"type": "step", **step})
                    messages.append({"role": "assistant", "content": text})
                    break
                # Keep the model's own words in the history: use structured tool_calls only when the
                # template renders them and the model wrote them in the template's own format.
                if structured_history and parsed["format"] == "tool_call tag":
                    messages.append({"role": "assistant", "content": parsed["content"],
                                     "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                                    for c in parsed["calls"]]})
                else:
                    messages.append({"role": "assistant", "content": text})
                results_msg = []
                for c in parsed["calls"]:
                    call = {"name": c["name"], "arguments": c["arguments"], "text": agent_bench.call_text(c["name"], c["arguments"]),
                            "env": env.owner(c["name"]), "unknown": env.owner(c["name"]) is None}
                    if bench.executes:
                        before = env.snapshot()
                        t0 = time.perf_counter()
                        result = env.call(c["name"], c["arguments"])
                        call.update(result=result, error=is_error(result), exec_ms=round(1000 * (time.perf_counter() - t0), 2),
                                    mutates=env.snapshot() != before)
                    else:
                        call.update(result=None, error=False, exec_ms=0, mutates=False)
                    call["action"] = action_letter(call, bench.executes)
                    step["calls"].append(call)
                    if call["result"] is not None:
                        turn_results.append(call["result"])
                    if native:
                        messages.append({"role": "tool", "name": c["name"], "content": call["result"] or ""})
                    else:
                        results_msg.append(f"<tool_response>\n{json.dumps({'name': c['name'], 'result': call['result']})}\n</tool_response>")
                if results_msg:
                    messages.append({"role": "user", "content": "\n".join(results_msg)})
                for text in {c["text"] for c in step["calls"]}:   # count steps, not duplicates within one reply
                    seen[text] = seen.get(text, 0) + 1
                steps.append(step)
                emit({"type": "step", **step})
                if not bench.executes:
                    turn_end = "called"
                    break
                if loop_guard and max(seen.values()) >= loop_guard:
                    turn_end = "repeat_loop"   # same call made loop_guard times in this turn: end it early
                    break
            else:
                turn_end = "step_limit"
            turns.append({"turn": ti, "end": turn_end, "results": turn_results, "state": env.snapshot()})
            emit({"type": "turn_end", "turn": ti, "end": turn_end})
            if stopped_reason:
                break
    finally:
        if hasattr(env, "close"):
            env.close()

    grade = bench.grade(task, turns, steps)
    turn_info = [{"turn": t["turn"], "end": t["end"], "n_results": len(t["results"])} for t in turns]
    flags = detect(steps, turn_info, grade)
    return {
        "benchmark": bench.key, "native_tools": native, "structured_history": structured_history, "steps": steps, "grade": grade, "stopped": stopped_reason,
        "turns": turn_info, "flags": flags, "failure": classify(grade, steps, turn_info, stopped_reason),
        "actions": action_string(steps), "loop_guard": loop_guard,
        "final_state": agent_bench.to_jsonable(env.snapshot()) if bench.executes else None,
        "totals": {**totals(steps, time.perf_counter() - t_start), "gt_calls": bench.ref_calls(task),
                   "max_repeat": max_repeat(steps)},
    }


# ---------------------------------------------------------------- trajectory analysis
# Ideas from TraceHarness (github.com/aura-lab-wm/traceharness, MIT): an action string per
# trajectory, repeated-call detection (threshold 3), step-level detector flags, and one
# failure mode per run. Letters here describe tool use rather than code edits:
#   R read (state unchanged) · W write (changed the environment) · E tool error
#   U unknown / hallucinated tool · A plain-text answer (ends the turn) · X unparseable output
#   C call that the benchmark checks without executing it (BFCL single-turn)
REPEAT_THRESHOLD = 3


def action_letter(call: dict, executed: bool = True) -> str:
    if call["unknown"]:
        return "U"
    if not executed:
        return "C"
    return "E" if call["error"] else "W" if call["mutates"] else "R"


def action_string(steps: list) -> str:
    out = []
    for s in steps:
        if s["calls"]:
            out += [c["action"] for c in s["calls"]]
        else:
            out.append("X" if s["parse_errors"] else "A")
    return "".join(out)


def max_repeat(steps: list) -> int:
    counts = {}
    for s in steps:
        for c in s["calls"]:
            key = (s["turn"], c["text"])
            counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def detect(steps: list, turns: list, grade: dict) -> list:
    """Step-level findings, each pointing at the step where it fired."""
    flags, seen, failing = [], {}, {}

    def flag(kind, s, detail, severity="medium"):
        flags.append({"kind": kind, "turn": s["turn"], "step": s["step"], "detail": detail, "severity": severity})

    for s in steps:
        if s["parse_errors"]:
            flag("format_error", s, s["parse_errors"][0][:160], "high")
        for c in s["calls"]:
            key = (s["turn"], c["text"])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == REPEAT_THRESHOLD:
                flag("repeat_loop", s, f"{c['text'][:120]} called {REPEAT_THRESHOLD} times in this turn", "high")
            if c["unknown"]:
                flag("unknown_tool", s, f"{c['name']} is not one of the available tools", "high")
            elif c["error"]:
                failing[key] = failing.get(key, 0) + 1
                if failing[key] == 2:
                    flag("error_persists", s, f"{c['text'][:120]} failed again with the same arguments")
    for check in grade["turns"]:
        if check["error_type"] == "empty_turn" and not check["after_failure"]:
            first = next((s for s in steps if s["turn"] == check["turn"]), None)
            if first:
                flag("answered_without_acting", first, "Replied in text although this turn needs tool calls", "high")
    for t in turns:
        mine = [s for s in steps if s["turn"] == t["turn"]]
        if t["end"] == "step_limit" and mine:
            flag("step_limit", mine[-1], "Hit the per-turn step limit", "high")
        if t["end"] == "repeat_loop" and mine:
            flag("loop_guard", mine[-1], "Turn ended by the loop guard: the same call was repeated", "high")
    return flags


def classify(grade: dict, steps: list, turns: list, stopped) -> str:
    """One failure mode per run, decided at the first failing turn."""
    if grade["valid"]:
        return "passed"
    if stopped:
        return stopped   # "stopped" by the user, or "context_full"
    err = grade["first_error"]["error_type"] or ""
    single = {"called_irrelevant": "called_when_irrelevant", "no_call": "no_tool_call"}
    if err in single:
        return single[err]
    if "wrong_func_name" in err:
        return "wrong_function"
    if "wrong_count" in err:
        return "wrong_call_count"
    if any(k in err for k in ("missing_required", "unexpected_param", "missing_optional")):
        return "wrong_parameters"
    if err.startswith(("type_error", "value_error")) or err in ("checker_error", "ast_mismatch") or "cannot_find" in err:
        return "wrong_values"
    if ":" in err:   # any other BFCL single-turn checker verdict ("<checker>:<reason>")
        return "wrong_values"
    turn = grade["first_error"]["turn"]
    mine = [s for s in steps if s["turn"] == turn]
    calls = [c for s in mine for c in s["calls"]]
    end = next((t["end"] for t in turns if t["turn"] == turn), None)
    if end in ("step_limit", "repeat_loop"):
        return end
    if err == "no_answer":
        return "step_limit" if end == "step_limit" else "no_answer"
    if any(c["unknown"] for c in calls):
        return "hallucinated_tool"
    if not calls:
        return "format_error" if any(s["parse_errors"] for s in mine) else "no_tool_call"
    if err == "response_mismatch":
        return "wrong_result"
    if err == "wrong_answer":
        return "wrong_answer"
    return "tool_errors" if any(c["error"] for c in calls) else "wrong_state"


def is_error(result: str) -> bool:
    # Exceptions come back as "Error during execution: ..."; the tools themselves report {"error": "..."}.
    if result.startswith(("Error during execution", "an error occured when call")):
        return True
    if result.startswith("{"):
        try:
            return "error" in json.loads(result)
        except ValueError:
            return False
    return False


def totals(steps: list, wall: float) -> dict:
    calls = [c for s in steps for c in s["calls"]]
    gen_s = sum(s["gen_s"] for s in steps)
    out_tok = sum(s["new_tokens"] for s in steps)
    return {"steps": len(steps), "calls": len(calls), "errors": sum(c["error"] for c in calls),
            "unknown_tools": sum(c["unknown"] for c in calls), "parse_failures": sum(bool(s["parse_errors"]) for s in steps),
            "prompt_tokens": sum(s["prompt_tokens"] for s in steps), "new_tokens": out_tok,
            "gen_s": round(gen_s, 2), "wall_s": round(wall, 2), "decode_tps": round(out_tok / gen_s, 2) if gen_s else None}


def _render(tok, messages, tools):
    if tok.chat_template:
        kw = {"tools": tools} if tools else {}
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"


def _generate(model, tok, inputs, max_new, opts, stop_flag, no_grad, on_token):
    streamer = _Streamer(tok)
    kwargs = dict(**inputs, max_new_tokens=max_new, streamer=streamer, stopping_criteria=StoppingCriteriaList([_Stop(stop_flag)]),
                  pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    temp = float(opts.get("temperature", 0))
    kwargs.update(dict(do_sample=True, temperature=temp, top_p=0.95) if temp > 0 else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
    out, err = {}, []

    def work():
        try:
            with no_grad:
                out["ids"] = model.generate(**kwargs)
        except Exception as e:  # noqa: BLE001
            err.append(e)
            streamer.end()

    t0 = time.perf_counter()
    th = threading.Thread(target=work, daemon=True)
    th.start()
    for chunk in streamer:
        on_token(chunk, streamer.n_tokens)
    th.join()
    if err:
        raise err[0]
    t1 = time.perf_counter()
    n_prompt = inputs["input_ids"].shape[1]
    new = out["ids"][0][n_prompt:]
    text = tok.decode(new, skip_special_tokens=True)
    first = streamer.first or t1
    n = streamer.n_tokens
    return text, {"new_tokens": n, "ttft_s": round(first - t0, 4), "gen_s": round(t1 - t0, 3),
                  "decode_tps": round((n - 1) / (t1 - first), 2) if n > 1 and t1 > first else None,
                  "hit_max_tokens": n >= max_new}
