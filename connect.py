"""Start a QuantBench backend and connect it to the hosted website.

Runs on Google Colab, a lab GPU server, or your own laptop / desktop (macOS, Linux, Windows).
An NVIDIA GPU gives full speed and every quantization method; Apple Silicon and CPU-only
machines run small models with optimum-quanto INT8 / INT4.

  Colab:          !curl -fsSL https://<site>/backend.tar.gz | tar xz -C /content
                  %run /content/quantbench/connect.py --site https://<site>
  This computer:  python quantbench/connect.py --site https://<site> --local
  Remote server:  python quantbench/connect.py --site https://<site>

It installs missing packages and starts server.py on 127.0.0.1 with a random access key.
With --local that server also serves the website itself (same login and shared history), so the
browser on this computer needs no tunnel and no cross-site permission. Otherwise a Cloudflare
quick tunnel (no account needed) gives it a temporary public HTTPS URL. It then prints (and with
--local opens) a link that loads the website already connected to this backend. Only someone with
that link (it holds the key) can use this backend. Running it again replaces the previous backend.
"""
import argparse
import importlib.util
import json
import os
import platform
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
import tarfile
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_DIR = Path.home() / ".cache" / "quantbench"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
BCB_REQUIREMENTS = "https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
# import name -> pip package
PACKAGES = {"torch": "torch", "transformers": "transformers", "accelerate": "accelerate",
            "fastapi": "fastapi", "uvicorn": "uvicorn", "pandas": "pandas", "pyarrow": "pyarrow",
            "huggingface_hub": "huggingface_hub", "ruff": "ruff", "pylint": "pylint", "radon": "radon", "bandit": "bandit",
            # agent page: BFCL's math tools, and modules ToolHop's tools import
            "mpmath": "mpmath", "pytz": "pytz", "dateutil": "python-dateutil", "babel": "babel", "dicttoxml": "dicttoxml"}
IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content") and os.environ.get("COLAB_RELEASE_TAG")
WINDOWS = os.name == "nt"


def log(msg):
    print(f"[quantbench] {msg}", flush=True)


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:   # parent package missing
        return False


def install_missing(with_bcb: bool):
    missing = [pkg for mod, pkg in PACKAGES.items() if not has_module(mod)]
    if missing:
        log(f"Installing {', '.join(missing)} (the first time can take several minutes) …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    # Quantization backends: bitsandbytes needs an NVIDIA GPU; optimum-quanto also runs on CPUs and Apple GPUs.
    probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
                           capture_output=True, text=True, check=True).stdout.split()
    torch_version, cuda = probe[0], probe[1] == "True"
    extra = "bitsandbytes" if cuda else "optimum.quanto"
    if not has_module(extra):
        pkg = "bitsandbytes" if cuda else "optimum-quanto"
        log(f"Installing {pkg} …")
        # Pin torch so pip picks a compatible version instead of replacing torch.
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg, f"torch=={torch_version}"]).returncode:
            log(f"Could not install {pkg}; its quantization options will be unavailable.")
    if with_bcb:
        log("Installing BigCodeBench test dependencies (a few minutes) …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", BCB_REQUIREMENTS], check=False)


def cloudflared_binary() -> str:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / ("cloudflared.exe" if WINDOWS else "cloudflared")
    if path.exists():
        return str(path)
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    system = platform.system()
    log("Downloading cloudflared …")
    if system == "Darwin":   # macOS builds ship as .tgz
        archive = STATE_DIR / "cloudflared.tgz"
        urllib.request.urlretrieve(f"{CLOUDFLARED_URL}cloudflared-darwin-{arch}.tgz", archive)
        with tarfile.open(archive) as tar:
            member = tar.getmember("cloudflared")
            member.name = path.name
            tar.extract(member, STATE_DIR)
        archive.unlink()
    elif WINDOWS:
        urllib.request.urlretrieve(f"{CLOUDFLARED_URL}cloudflared-windows-{arch}.exe", path)
    else:
        urllib.request.urlretrieve(f"{CLOUDFLARED_URL}cloudflared-linux-{arch}", path)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def spawn(cmd, log_file, **kw):
    """Start a process in its own group so it outlives this script (Colab) and can be stopped as a unit."""
    if WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, **kw)


def kill(pid: int):
    try:
        if WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def stop_previous():
    pid_file = STATE_DIR / "pids.json"
    if not pid_file.exists():
        return
    for pid in json.loads(pid_file.read_text()).values():
        kill(pid)
    pid_file.unlink()
    time.sleep(2)


def wait_for(url: str, timeout: float, what: str):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if json.load(r).get("app") == "quantbench":
                    return
        except Exception as e:  # noqa: BLE001 - keep retrying until the deadline
            last = e
        time.sleep(1.5)
    raise RuntimeError(f"{what} did not come up within {timeout:.0f}s ({last})")


def start(args):
    if args.local and IN_COLAB:
        raise SystemExit("--local is for running on your own computer; on Colab leave it out (the tunnel is needed).")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stop_previous()
    install_missing(args.with_bigcodebench_deps)
    key = secrets.token_urlsafe(24)

    server_log = open(STATE_DIR / "server.log", "w")
    cmd = [sys.executable, str(HERE / "server.py"), "--host", "127.0.0.1", "--port", str(args.port),
           "--auth-key", key, "--max-download-gb", str(args.max_download_gb), "--db", str(STATE_DIR / "history.sqlite")]
    if args.model:
        cmd += ["--model", args.model]
    if args.local:
        cmd += ["--site-config", str(site_config(args.site))]
    log("Starting the backend (first start imports torch, ~20 s) …")
    server = spawn(cmd, server_log, cwd=HERE)
    pids = {"server": server.pid}
    (STATE_DIR / "pids.json").write_text(json.dumps(pids))
    try:
        wait_for(f"http://127.0.0.1:{args.port}/api/health", 180, "The backend")
    except RuntimeError as e:
        raise RuntimeError(f"{e}. Last lines of {STATE_DIR / 'server.log'}:\n" +
                           "\n".join((STATE_DIR / "server.log").read_text().splitlines()[-15:])) from None

    if args.local:
        local = f"http://127.0.0.1:{args.port}"
        return finish(args, local, key, pids, page=local)

    tunnel_log_path = STATE_DIR / "tunnel.log"
    tunnel_log = open(tunnel_log_path, "w")
    log("Opening a Cloudflare quick tunnel …")
    tunnel = spawn([cloudflared_binary(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{args.port}"], tunnel_log)
    pids["tunnel"] = tunnel.pid
    (STATE_DIR / "pids.json").write_text(json.dumps(pids))

    public = None
    deadline = time.time() + 60
    while time.time() < deadline and public is None:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", tunnel_log_path.read_text())
        public = m.group(0) if m else None
        time.sleep(1)
    if public is None:
        raise RuntimeError(f"No tunnel URL; see {tunnel_log_path}")
    wait_for(f"{public}/api/health", 90, "The tunnel")
    return finish(args, public, key, pids)


def site_config(site: str) -> Path:
    """The hosted site's public config (Supabase URL + anon key), so the local copy of the page uses the same login and history."""
    with urllib.request.urlopen(f"{site.rstrip('/')}/config.js", timeout=30) as r:
        text = r.read().decode()
    m = re.search(r"window\.QB_CONFIG\s*=\s*(\{.*\})\s*;?\s*$", text, re.S)
    config = json.loads(m.group(1)) if m else {}
    if config.get("mode") != "hosted":
        raise RuntimeError(f"{site} does not look like a hosted QuantBench site (no config.js)")
    config["site"] = site.rstrip("/")
    path = STATE_DIR / "site_config.json"
    path.write_text(json.dumps(config))
    return path


def finish(args, backend: str, key: str, pids: dict, page: str | None = None):
    link = f"{(page or args.site).rstrip('/')}/#backend={backend}&key={key}"
    log(f"Backend ready: {backend}")
    log(f"Logs: {STATE_DIR / 'server.log'}")
    where = "this computer" if args.local else "this machine"
    print(f"\nOpen QuantBench connected to {where}:\n{link}\n\n"
          "Anyone with this link can use this backend, so don't share it.", flush=True)
    if args.local:
        print("This copy of the site is served by your computer; it only opens in a browser on this computer.\n"
              "Sign in with the same account as on the website; runs go to the same shared history.", flush=True)
    if IN_COLAB:
        from IPython.display import HTML, display
        display(HTML(f'<a href="{link}" target="_blank" style="display:inline-block;padding:10px 18px;background:#238636;'
                     f'color:#fff;border-radius:8px;font:600 14px sans-serif;text-decoration:none">Open QuantBench ▶</a>'))
        print("Keep this runtime running while you use the site.", flush=True)
        return link
    if args.local and not args.no_browser:
        webbrowser.open(link)
    if args.detach:
        print(f"\nRunning in the background. Stop it with: python {Path(__file__).name} --stop\n", flush=True)
        return link
    print("\nKeep this terminal open while you use the site. Press Ctrl+C to stop.\n", flush=True)
    try:
        while all(p is None for p in poll_all(pids)):
            time.sleep(2)
        log(f"The backend stopped unexpectedly; see {STATE_DIR / 'server.log'}")
    except KeyboardInterrupt:
        pass
    stop_previous()
    log("Stopped.")
    return link


def poll_all(pids: dict):
    """Exit codes of our processes (None while running)."""
    out = []
    for pid in pids.values():
        try:
            if WINDOWS:
                alive = str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
            else:
                os.kill(pid, 0)
                alive = os.waitpid(pid, os.WNOHANG) == (0, 0)
        except (ProcessLookupError, ChildProcessError):
            alive = False
        out.append(None if alive else 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", required=True, help="URL of the hosted QuantBench website")
    p.add_argument("--local", action="store_true",
                   help="Use the backend from a browser on this same computer (no tunnel); opens the site for you")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default=None, help="Default model shown in the UI")
    p.add_argument("--max-download-gb", type=float, default=40, help="Refuse larger models (protects the runtime's disk)")
    p.add_argument("--with-bigcodebench-deps", action="store_true",
                   help="Also install BigCodeBench's test dependencies (needed for full BigCodeBench grading; may change package versions)")
    p.add_argument("--detach", action="store_true", help="Keep running in the background after this script exits")
    p.add_argument("--no-browser", action="store_true", help="With --local, print the link instead of opening it")
    p.add_argument("--stop", action="store_true", help="Stop the backend and tunnel started earlier")
    args = p.parse_args()
    if args.stop:
        stop_previous()
        log("Stopped.")
        return
    start(args)


if __name__ == "__main__":
    main()
