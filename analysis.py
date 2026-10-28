"""Static analysis of generated code: ruff (lint), pylint (score), radon (complexity), bandit (security).

The code is only parsed, never executed, so this is safe to run on any model output.
"""
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from radon.complexity import cc_rank, cc_visit
from radon.metrics import mi_rank, mi_visit
from radon.raw import analyze as raw_analyze

TOOL_TIMEOUT = 30
RUFF_RULES = "E,W,F,B,SIM,C4,N"  # pycodestyle, pyflakes, bugbear, simplify, comprehensions, naming (no pyupgrade: benchmarks dictate typing.List-style signatures)
MAX_ISSUES = 40


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TOOL_TIMEOUT)


def ruff(path: Path) -> dict:
    proc = _run([sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache", "--output-format=json",
                 "--select", RUFF_RULES, "--line-length", "100", "--target-version", "py310", str(path)])
    items = json.loads(proc.stdout or "[]")
    issues = [{"code": i["code"] or "syntax", "message": i["message"], "line": i["location"]["row"]} for i in items]
    return {"count": len(issues), "syntax_error": any(i["code"] == "invalid-syntax" for i in issues),
            "issues": issues[:MAX_ISSUES]}


def pylint(path: Path) -> dict:
    proc = _run([sys.executable, "-m", "pylint", "--output-format=json2", "--persistent=n",
                 "--disable=missing-module-docstring", str(path)])
    data = json.loads(proc.stdout)
    msgs = [{"code": m["messageId"], "symbol": m["symbol"], "type": m["type"], "message": m["message"],
             "line": m["line"]} for m in data["messages"]]
    score = data["statistics"].get("score")
    return {"score": round(max(score, 0.0), 2) if score is not None else 0.0,
            "counts": data["statistics"]["messageTypeCount"], "issues": msgs[:MAX_ISSUES]}


def bandit(path: Path) -> dict:
    proc = _run([sys.executable, "-m", "bandit", "-f", "json", "-q", str(path)])
    data = json.loads(proc.stdout)
    res = data.get("results", [])
    sev = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in res:
        sev[r["issue_severity"]] = sev.get(r["issue_severity"], 0) + 1
    return {"count": len(res), "high": sev["HIGH"], "medium": sev["MEDIUM"], "low": sev["LOW"],
            "issues": [{"code": r["test_id"], "severity": r["issue_severity"], "message": r["issue_text"],
                        "line": r["line_number"]} for r in res][:MAX_ISSUES]}


def radon(code: str) -> dict:
    blocks = cc_visit(code)
    ccs = [b.complexity for b in blocks]
    raw = raw_analyze(code)
    mi = mi_visit(code, multi=True)
    avg = sum(ccs) / len(ccs) if ccs else 0.0
    return {"cc_avg": round(avg, 2), "cc_max": max(ccs, default=0), "cc_rank": cc_rank(avg) if ccs else "–",
            "functions": len(blocks), "mi": round(mi, 1), "mi_rank": mi_rank(mi),
            "sloc": raw.sloc, "comments": raw.comments + raw.multi}


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 - one tool failing (e.g. syntax error) shouldn't hide the others
        return {"error": f"{type(e).__name__}: {e}"}


def analyze(code: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "candidate.py"
        path.write_text(code)
        with ThreadPoolExecutor(3) as pool:
            jobs = {name: pool.submit(_safe, fn, path) for name, fn in (("ruff", ruff), ("pylint", pylint), ("bandit", bandit))}
            result = {"radon": _safe(radon, code)}
            result.update({name: job.result() for name, job in jobs.items()})
    result["lines"] = code.count("\n") + 1
    return result
