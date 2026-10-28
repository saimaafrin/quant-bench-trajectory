"""Turn a local model's raw text into tool calls.

Models with a tool-aware chat template (Qwen2.5, Qwen3, Hammer, xLAM, ...) get the tools
through `apply_chat_template(tools=...)` and answer with Hermes-style blocks:
    <tool_call>{"name": "mv", "arguments": {"source": "a", "destination": "b"}}</tool_call>
Other models get the same instructions in the system prompt (FALLBACK_PROMPT). The parser also
accepts the common near-misses small models produce (JSON lists, a fenced json block, a bare
{"name": ..., "arguments": ...} object, BFCL's python-call style), so a quantized model is
not penalised for formatting the original happened to get right: the trajectory records which
form was used.
"""
import ast
import json
import re

TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
PY_LITERALS = {"true": "True", "false": "False", "null": "None"}
FENCE = re.compile(r"```(?:json|tool_call|python)?\s*(.*?)```", re.S)

FALLBACK_PROMPT = """You can call these tools (JSON schemas):
{tools}

To call a tool, reply with one block per call, nothing else:
<tool_call>{{"name": "<tool name>", "arguments": {{<arguments as JSON>}}}}</tool_call>
When the request is fully handled, reply in plain text without any tool call."""


def parse(text: str) -> dict:
    """-> {"calls": [{"name", "arguments"}], "format": str, "content": text outside calls, "errors": [...]}"""
    calls, errors = [], []
    blocks = TOOL_BLOCK.findall(text)
    if blocks:
        for b in blocks:
            got, err = _from_json(b)
            calls += got
            if err:
                errors.append(err)
        return {"calls": calls, "format": "tool_call tag", "content": TOOL_BLOCK.sub("", text).strip(), "errors": errors}

    # Some models write one fenced block per call ("```[...]``` ```[...]```"): collect them all.
    fenced = []
    for body in FENCE.findall(text):
        stripped = body.strip()
        if stripped[:1] in "[{":
            got, err = _from_json(stripped)
            fenced += got
            if err and '"name"' in stripped.replace("'", '"'):
                errors.append(err)
    if fenced:
        return {"calls": fenced, "format": "fenced block", "content": FENCE.sub("", text).strip(), "errors": errors}
    stripped = text.strip()
    if stripped[:1] in "[{":
        got, err = _from_json(stripped)
        if got:
            return {"calls": got, "format": "bare JSON", "content": "", "errors": []}
        if err and '"name"' in stripped:
            errors.append(err)

    got = _from_python(text.strip())   # BFCL default style: [mv(source='a'), cd(folder='b')]
    if got:
        return {"calls": got, "format": "python call", "content": "", "errors": []}
    return {"calls": [], "format": None, "content": text.strip(), "errors": errors}


def _from_json(s: str):
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Small models often emit Python literals (single quotes, True/None) inside the JSON.
        try:
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            try:   # ...or a mix of both: {'a': true}
                obj = ast.literal_eval(re.sub(r"\b(true|false|null)\b", lambda m: PY_LITERALS[m.group(1)], s))
            except (ValueError, SyntaxError):
                return [], f"unparseable tool call: {s[:200]}"
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for it in items:
        if not isinstance(it, dict):
            return calls, f"tool call is not an object: {str(it)[:200]}"
        name = it.get("name") or it.get("function") or (it.get("function_call") or {}).get("name")
        if isinstance(name, dict):   # {"function": {"name": ..., "arguments": ...}}
            it, name = name, name.get("name")
        args = it.get("arguments", it.get("parameters", it.get("args", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return calls, f"arguments are not JSON: {args[:200]}"
        if not name:
            return calls, f"tool call without a name: {str(it)[:200]}"
        calls.append({"name": str(name), "arguments": args if args is not None else {}})
    return calls, None


def _from_python(s: str):
    s = s.strip().strip("`").strip()
    if not s:
        return []
    try:
        node = ast.parse(s if s.startswith("[") else f"[{s}]", mode="eval").body
    except SyntaxError:
        return []
    if not isinstance(node, ast.List) or not node.elts:
        return []
    calls = []
    for c in node.elts:
        if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)) or c.args:
            return []
        try:
            calls.append({"name": c.func.id, "arguments": {k.arg: ast.literal_eval(k.value) for k in c.keywords}})
        except ValueError:
            return []
    return calls
