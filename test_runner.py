"""Runs model-generated code plus tests in a separate process.

Usage: python -I test_runner.py payload.json
payload: {"code": str, "tests": [str], "kind": "asserts" | "unittest"}
  asserts : each test is a statement block exec'd after the code (built-in tasks, HumanEval check()).
  unittest: tests[0] defines unittest.TestCase classes (BigCodeBench); every test method is reported.
Prints one line: __RESULT__{json}
"""
import contextlib
import io
import json
import os
import sys
import traceback
import types
import unittest


def _last_line(exc_text: str) -> str:
    lines = [l for l in exc_text.strip().splitlines() if l.strip()]
    return lines[-1][:300] if lines else "error"


def run_asserts(ns, tests):
    out = []
    for test in tests:
        try:
            exec(test, ns)
            out.append({"test": test, "ok": True})
        except BaseException as e:  # noqa: BLE001
            out.append({"test": test, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
    return out


def run_unittest(ns, tests):
    before = {k: id(v) for k, v in ns.items()}
    exec(tests[0], ns)
    cases = [v for k, v in ns.items() if before.get(k) != id(v)
             and isinstance(v, type) and issubclass(v, unittest.TestCase)]
    out = []
    loader = unittest.TestLoader()
    for cls in cases:
        for name in loader.getTestCaseNames(cls):
            result = unittest.TestResult()
            cls(name).run(result)
            problems = result.failures + result.errors
            entry = {"test": f"{cls.__name__}.{name}", "ok": not problems}
            if problems:
                entry["error"] = _last_line(problems[0][1])
            out.append(entry)
    return out


def main():
    with open(sys.argv[1]) as f:
        payload = json.load(f)
    # Mirror BigCodeBench's harness: code + tests live in a real module registered in sys.modules,
    # so tests that patch or import the candidate module by name work.
    module = types.ModuleType("__test__")
    module.__dict__.update({"__file__": "__test__.py", "sys": sys, "os": os})
    sys.modules["__test__"] = module
    ns = module.__dict__
    res = {"compiled": True, "error": None, "results": []}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            exec(compile(payload["code"], "<candidate>", "exec"), ns)
        except BaseException as e:  # noqa: BLE001 - report anything the candidate raises
            res["compiled"] = False
            res["error"] = f"{type(e).__name__}: {e}"[:500]
        else:
            try:
                runner = run_unittest if payload.get("kind") == "unittest" else run_asserts
                res["results"] = runner(ns, payload["tests"])
            except BaseException:  # noqa: BLE001 - e.g. the test module itself fails to import
                res["error"] = _last_line(traceback.format_exc())
    print("__RESULT__" + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
