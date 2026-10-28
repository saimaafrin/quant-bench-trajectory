"""Persistent run history (SQLite).

Every generated problem is stored as one row (full record + compact summary). Rows that belong to
the same batch share a batch_id; a single Generate is a run of one problem. The UI groups problems
into runs and aggregates them (static/index.html), for this store and for the hosted Supabase one.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SUMMARY_VERSION = 2   # bump when summarize() changes so stored summaries get recomputed
SLOTS = ("a", "b")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created  TEXT NOT NULL,
    batch_id TEXT,
    source   TEXT,
    task     TEXT,
    summary  TEXT NOT NULL,
    record   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_batch ON runs(batch_id);
"""


class RunStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
        self._refresh_summaries()

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _refresh_summaries(self):
        with self.lock, self._conn() as c:
            stale = [(r["id"], r["record"]) for r in c.execute("SELECT id, summary, record FROM runs")
                     if json.loads(r["summary"]).get("summary_version") != SUMMARY_VERSION]
            for row_id, record in stale:
                c.execute("UPDATE runs SET summary = ? WHERE id = ?", (json.dumps(summarize(json.loads(record))), row_id))

    # ------------------------------------------------------------ problems (rows)
    def add(self, record: dict) -> dict:
        record = dict(record)
        record["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        summary = summarize(record)
        with self.lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs (created, batch_id, source, task, summary, record) VALUES (?, ?, ?, ?, ?, ?)",
                (record["created"], record.get("batch_id"), record.get("source"), record.get("task"),
                 json.dumps(summary), json.dumps(record)))
        return {"id": cur.lastrowid, "created": record["created"], "group": group_key(record.get("batch_id"), cur.lastrowid)}

    def get(self, row_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT id, record FROM runs WHERE id = ?", (row_id,)).fetchone()
        return {"id": row["id"], **json.loads(row["record"])} if row else None

    # ------------------------------------------------------------ runs (groups of problems)
    def _summaries(self, batch_id=None, row_id=None):
        sql, args = "SELECT id, created, summary FROM runs", ()
        if batch_id is not None:
            sql, args = sql + " WHERE batch_id = ?", (batch_id,)
        elif row_id is not None:
            sql, args = sql + " WHERE id = ? AND batch_id IS NULL", (row_id,)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY id", args).fetchall()
        return [{"id": r["id"], "created": r["created"], **json.loads(r["summary"])} for r in rows]

    def rows(self) -> list[dict]:
        """Summaries of every stored problem; the UI groups them into runs."""
        return self._summaries()

    def delete_group(self, key: str) -> int:
        ids = [r["id"] for r in self._rows_for(key)]
        with self.lock, self._conn() as c:
            c.executemany("DELETE FROM runs WHERE id = ?", [(i,) for i in ids])
        return len(ids)

    def _rows_for(self, key: str):
        if key.startswith("single-"):
            return self._summaries(row_id=int(key.split("-", 1)[1]))
        return self._summaries(batch_id=key)


def group_key(batch_id, row_id):
    return batch_id or f"single-{row_id}"


# ---------------------------------------------------------------- per-problem summary
def summarize(rec: dict) -> dict:
    def slot(s):
        model, res = (rec.get("models") or {}).get(s), (rec.get("slots") or {}).get(s)
        if not model or not res:
            return None
        m, t, q = res.get("metrics") or {}, res.get("tests"), res.get("quality") or {}
        pl, rf, rd, bd = (q.get(k) or {} for k in ("pylint", "ruff", "radon", "bandit"))
        ran = bool(t) and t.get("compiled", True) and not t.get("error") and not t.get("timeout")
        return {
            "repo": model.get("repo"), "precision": model.get("precision"), "label": model.get("label"),
            "footprint": (model.get("info") or {}).get("footprint"),
            "decode_tps": m.get("decode_tps"), "ttft_s": m.get("ttft_s"), "tokens": m.get("new_tokens"),
            "total_s": m.get("total_s"), "peak_mem": m.get("peak_mem"), "stopped": m.get("stopped"),
            "graded": t is not None,
            "pass": bool(t.get("all_passed")) if t else None,
            "ran": ran if t else None, "timeout": bool(t.get("timeout")) if t else None,
            "tests_passed": t.get("passed") if t else None, "tests_total": t.get("total") if t else None,
            "analyzed": bool(q),
            "pylint": pl.get("score") if q and "error" not in pl else None,
            "ruff": rf.get("count") if q and "error" not in rf else None,
            "syntax_error": bool(rf.get("syntax_error")) if q else None,
            "cc": rd.get("cc_avg") if q and "error" not in rd else None,
            "mi": rd.get("mi") if q and "error" not in rd else None,
            "sloc": rd.get("sloc") if q and "error" not in rd else None,
            "bandit": bd.get("count") if q and "error" not in bd else None,
            "bandit_high": bd.get("high"), "bandit_medium": bd.get("medium"),
            "issues": issue_counts(pl, rf, bd) if q else [],
        }

    sim = rec.get("similarity") or {}
    return {"summary_version": SUMMARY_VERSION, "source": rec.get("source"), "task": rec.get("task"),
            "task_index": rec.get("task_index"), "batch_id": rec.get("batch_id"), "batch_label": rec.get("batch_label"),
            "config": rec.get("config"), "a": slot("a"), "b": slot("b"), "similarity": sim.get("similarity")}


def issue_counts(pl, rf, bd):
    found = {}

    def add(tool, code, name):
        key = f"{tool}:{code}"
        found.setdefault(key, {"tool": tool, "code": code, "name": name, "n": 0})["n"] += 1

    for i in rf.get("issues", []):
        add("ruff", i["code"], i["message"])
    for i in pl.get("issues", []):
        add("pylint", i["code"], i["symbol"])
    for i in bd.get("issues", []):
        add("bandit", i["code"], f"{i['message']} ({i['severity'].lower()})")
    return list(found.values())


# ---------------------------------------------------------------- agent page (tool-calling trajectories)
AGENT_SUMMARY_VERSION = 1
AGENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created  TEXT NOT NULL,
    batch_id TEXT,
    task     TEXT,
    summary  TEXT NOT NULL,
    record   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_runs_batch ON agent_runs(batch_id);
"""


class AgentStore(RunStore):
    """Same database file, separate table, so the code page's history is untouched."""

    def __init__(self, path: Path):   # pylint: disable=super-init-not-called
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript(AGENT_SCHEMA)

    def add(self, record: dict) -> dict:
        record = dict(record)
        record["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        summary = summarize_agent(record)
        with self.lock, self._conn() as c:
            cur = c.execute("INSERT INTO agent_runs (created, batch_id, task, summary, record) VALUES (?, ?, ?, ?, ?)",
                            (record["created"], record.get("batch_id"), record.get("task_id"), json.dumps(summary), json.dumps(record)))
        return {"id": cur.lastrowid, "created": record["created"], "group": group_key(record.get("batch_id"), cur.lastrowid)}

    def get(self, row_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT id, record FROM agent_runs WHERE id = ?", (row_id,)).fetchone()
        return {"id": row["id"], **json.loads(row["record"])} if row else None

    def _summaries(self, batch_id=None, row_id=None):
        sql, args = "SELECT id, created, summary FROM agent_runs", ()
        if batch_id is not None:
            sql, args = sql + " WHERE batch_id = ?", (batch_id,)
        elif row_id is not None:
            sql, args = sql + " WHERE id = ? AND batch_id IS NULL", (row_id,)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY id", args).fetchall()
        return [{"id": r["id"], "created": r["created"], **json.loads(r["summary"])} for r in rows]

    def delete_group(self, key: str) -> int:
        ids = [r["id"] for r in self._rows_for(key)]
        with self.lock, self._conn() as c:
            c.executemany("DELETE FROM agent_runs WHERE id = ?", [(i,) for i in ids])
        return len(ids)


def summarize_agent(rec: dict) -> dict:
    """Compact per-slot metrics for the agent history table and run reports."""
    def slot(s):
        model, ep = (rec.get("models") or {}).get(s), (rec.get("slots") or {}).get(s)
        if not model or not ep or "grade" not in ep:
            return None
        g, t = ep["grade"], ep.get("totals") or {}
        return {"repo": model.get("repo"), "precision": model.get("precision"), "label": model.get("label"),
                "footprint": (model.get("info") or {}).get("footprint"),
                "valid": g["valid"], "turns_ok": g["turns_ok"], "turns_total": g["turns_total"],
                "first_error": g.get("first_error"), "failure": ep.get("failure"), "actions": ep.get("actions"),
                "flags": sorted({f["kind"] for f in ep.get("flags", [])}), "native_tools": ep.get("native_tools"),
                "answer": g.get("answer"),
                **{k: t.get(k) for k in ("steps", "calls", "errors", "unknown_tools", "parse_failures", "prompt_tokens",
                                         "new_tokens", "gen_s", "wall_s", "decode_tps", "gt_calls", "max_repeat")}}

    return {"summary_version": AGENT_SUMMARY_VERSION, "benchmark": rec.get("benchmark", "bfcl_multi"),
            "task_id": rec.get("task_id"), "task_index": rec.get("task_index"),
            "envs": rec.get("envs"), "turns": rec.get("n_turns"), "batch_id": rec.get("batch_id"),
            "batch_label": rec.get("batch_label"), "config": rec.get("config"), "a": slot("a"), "b": slot("b"),
            "first_divergence": rec.get("first_divergence")}
