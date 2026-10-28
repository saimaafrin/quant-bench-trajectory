"""Build the static website for Vercel into deploy/quantbench/.

  python deploy/build_site.py --supabase-url https://xyz.supabase.co --supabase-anon-key <anon key>
  npx vercel deploy deploy/quantbench --prod --scope <team>

Output: index.html (same page as the local app), config.js (Supabase settings: public values),
backend.tar.gz (what the Colab cell downloads), QuantBench_Colab.ipynb, vercel.json.
"""
import argparse
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deploy" / "quantbench"
BACKEND_FILES = ["server.py", "connect.py", "analysis.py", "benchmarks.py", "catalog.py", "presets.py",
                 "runstore.py", "test_runner.py", "static/index.html",
                 "agent_bench.py", "agent_benchmarks.py", "agent_runner.py", "tool_calls.py", "static/agent.html",
                 "bfcl", "toolhop"]


def backend_tarball(dest: Path):
    with tarfile.open(dest, "w:gz") as tar:
        for rel in BACKEND_FILES:
            tar.add(ROOT / rel, arcname=f"quantbench/{rel}",
                    filter=lambda info: None if "__pycache__" in info.name else info)


def notebook(site: str) -> dict:
    md = lambda text: {"cell_type": "markdown", "metadata": {}, "source": text}
    code = lambda text: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            md("# QuantBench Live: GPU backend\n\n"
               "1. **Runtime → Change runtime type → T4 GPU** (or a larger GPU).\n"
               "2. Run the cell below (about 1–2 minutes).\n"
               "3. Click **Open QuantBench**. The website connects to this notebook's GPU.\n\n"
               "Keep this tab open while you use the site. Anyone with the printed link can use this GPU, so don't share it."),
            code(f"!curl -fsSL {site}/backend.tar.gz | tar xz -C /content\n"
                 f"%run /content/quantbench/connect.py --site {site}"),
            md("Optional: full BigCodeBench grading needs its test libraries. Run this instead of the cell above "
               "(takes a few minutes and may change some package versions):"),
            code(f"!curl -fsSL {site}/backend.tar.gz | tar xz -C /content\n"
                 f"%run /content/quantbench/connect.py --site {site} --with-bigcodebench-deps"),
            md("Server log, if something goes wrong:"),
            code("!tail -n 50 ~/.cache/quantbench/server.log"),
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--supabase-url", required=True)
    p.add_argument("--supabase-anon-key", required=True, help="The public anon / publishable key (never the service_role key)")
    p.add_argument("--site", default="https://quantbench.vercel.app", help="Public URL of the site (used in the notebook)")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.iterdir():   # start clean, but keep .vercel (the link to the Vercel project)
        if f.name != ".vercel":
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    shutil.copy(ROOT / "static" / "index.html", OUT / "index.html")
    shutil.copy(ROOT / "static" / "agent.html", OUT / "agent.html")
    config = {"mode": "hosted", "supabaseUrl": args.supabase_url.rstrip("/"), "supabaseAnonKey": args.supabase_anon_key,
              "site": args.site.rstrip("/")}
    (OUT / "config.js").write_text(f"window.QB_CONFIG = {json.dumps(config)};\n")
    backend_tarball(OUT / "backend.tar.gz")
    (OUT / "QuantBench_Colab.ipynb").write_text(json.dumps(notebook(args.site.rstrip("/")), indent=1))
    (OUT / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,   # /agent serves agent.html, as on the local server
        "headers": [
            {"source": "/(config.js|backend.tar.gz|index.html|agent.html|agent|)", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
            {"source": "/(.*)", "headers": [{"key": "X-Content-Type-Options", "value": "nosniff"},
                                           {"key": "Referrer-Policy", "value": "no-referrer"}]},
        ]
    }, indent=2))
    print(f"Built {OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
