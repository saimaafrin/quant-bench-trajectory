"""Runs ToolHop's tool functions (dataset code) in a separate, isolated process.

Started by agent_bench.ToolSandbox with `python -I`, an empty working directory and no GPUs.
Reads one JSON request per line: {"source": <function source>, "name": ..., "args": {...}}.
Writes one JSON reply per line: {"ok": true, "result": ...} or {"ok": false, "error": "..."}.
Each call gets a fresh namespace, as in ToolHop's official evaluation.
"""
import contextlib
import io
import json
import sys


def main():
    out = sys.stdout   # replies only; anything the tools print is captured and dropped
    for line in sys.stdin:
        try:
            req = json.loads(line)
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(req["source"], namespace)  # noqa: S102 - the benchmark's tools are Python functions
                result = namespace[req["name"]](**req["args"])
            json.dumps(result, ensure_ascii=False)   # must be JSON-serialisable to go back to the model
            reply = {"ok": True, "result": result}
        except Exception as e:  # noqa: BLE001 - reported to the agent as the tool's error
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}" if not str(e) else str(e)}
        except BaseException as e:   # SystemExit / KeyboardInterrupt raised by tool code
            reply = {"ok": False, "error": f"{type(e).__name__} raised by the tool"}
        out.write(json.dumps(reply, ensure_ascii=False, default=str) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
