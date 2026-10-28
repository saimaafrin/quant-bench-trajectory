"""QuantBench Live: load a full-precision and a quantized LLM side by side and stream
both answers to the same coding prompt, with live speed / memory / correctness metrics.

Run:  python server.py --port 8000   then open http://localhost:8000
"""
import os
import platform
import sys

os.environ.setdefault("USE_TF", "0")  # transformers would otherwise import TensorFlow if installed
# Tools installed next to this Python (e.g. ninja for quanto's INT4 kernels) even when the venv is not activated.
os.environ["PATH"] = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + os.environ.get("PATH", "")

import argparse
import difflib
import fnmatch
import gc
import importlib.util
import json
import queue
import secrets
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import disable_progress_bars
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

import agent_bench
import agent_benchmarks
import agent_runner
import analysis
import benchmarks
from catalog import agent_catalog, model_catalog
from runstore import AgentStore, RunStore, summarize, summarize_agent
from presets import PRESETS

ROOT = Path(__file__).resolve().parent
TEST_RUNNER = ROOT / "test_runner.py"

PRECISIONS = {
    "fp16": "FP16 (unquantized)",
    "bf16": "BF16 (unquantized)",
    "fp32": "FP32 (unquantized)",
    "bnb-8bit": "INT8 · bitsandbytes LLM.int8()",
    "bnb-4bit": "4-bit NF4 · bitsandbytes",
    "quanto-int8": "INT8 weights · optimum-quanto",
    "quanto-int4": "INT4 weights · optimum-quanto",
    "prequantized": "Pre-quantized repo (GPTQ / AWQ / ...)",
}
META_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken"]
LIMITS = {"max_download_bytes": None}  # set from --max-download-gb
DOWNLOAD_LOCK = threading.Lock()  # concurrent hub downloads in one process race (tqdm, shared blobs)
disable_progress_bars()  # the UI shows its own progress

import transformers as _transformers

VERSIONS = {"torch": torch.__version__, "transformers": _transformers.__version__, "python": platform.python_version()}
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = not HAS_CUDA and torch.backends.mps.is_available()   # Apple Silicon GPU
SMALL_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"   # default without an NVIDIA GPU


def precision_problems() -> dict:
    """Precisions this machine cannot load, with the reason shown in the UI."""
    bad = {}
    for p in ("bnb-8bit", "bnb-4bit"):
        if not HAS_CUDA:
            bad[p] = "needs an NVIDIA GPU"
        elif importlib.util.find_spec("bitsandbytes") is None:
            bad[p] = "needs bitsandbytes"
    for p in ("quanto-int8", "quanto-int4"):
        if importlib.util.find_spec("optimum") is None or importlib.util.find_spec("optimum.quanto") is None:
            bad[p] = "needs optimum-quanto"
    return bad


def ram_total():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):   # not available on Windows
        return None


def accelerator() -> str:
    if HAS_CUDA:
        return ", ".join(f"{g['name']} ({g['total'] / 1e9:.0f} GB)" for g in gpu_list())
    ram = ram_total()
    mem = f", {ram / 1e9:.0f} GB RAM" if ram else ""
    if HAS_MPS:
        return f"Apple GPU (MPS){mem}"
    return f"CPU only ({os.cpu_count()} cores{mem})"


def empty_cache():
    if HAS_CUDA:
        torch.cuda.empty_cache()
    elif HAS_MPS:
        torch.mps.empty_cache()


def device_alloc(device: str):
    """Memory currently held on a GPU device (None on CPU)."""
    idx = cuda_index(device)
    if idx is not None:
        return torch.cuda.memory_allocated(idx)
    if device == "mps":
        return torch.mps.current_allocated_memory()
    return None


try:
    import pynvml

    pynvml.nvmlInit()
except Exception:  # noqa: BLE001 - GPU stats are optional
    pynvml = None


def gpu_list():
    gpus = []
    for i in range(torch.cuda.device_count()):
        g = {"index": i, "name": torch.cuda.get_device_properties(i).name,
             "total": torch.cuda.get_device_properties(i).total_memory, "used": None}
        if pynvml is not None:
            try:
                # CUDA_VISIBLE_DEVICES remaps indices; fall back to the same index otherwise.
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                phys = int(visible.split(",")[i]) if visible else i
                mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(phys))
                g["used"], g["total"] = mem.used, mem.total
            except Exception:  # noqa: BLE001
                pass
        if g["used"] is None:
            try:
                free, total = torch.cuda.mem_get_info(i)
                g["used"], g["total"] = total - free, total
            except Exception:  # noqa: BLE001
                pass
        g["device"] = f"cuda:{i}"
        gpus.append(g)
    if HAS_MPS:   # unified memory: the GPU shares system RAM
        gpus.append({"index": 0, "device": "mps", "name": "Apple GPU", "total": ram_total(),
                     "used": torch.mps.driver_allocated_memory()})
    return gpus


def cuda_index(device: str):
    if not device.startswith("cuda"):
        return None
    return torch.device(device).index or 0


def tensor_bytes(t) -> int:
    # quanto wraps packed integer weights in a float-looking tensor; count the storage inside it.
    inner = [getattr(t, a) for a in ("_data", "_scale", "_shift") if isinstance(getattr(t, a, None), torch.Tensor)]
    return sum(tensor_bytes(x) for x in inner) if inner else t.numel() * t.element_size()


def footprint(model, precision: str) -> int:
    if not precision.startswith("quanto"):
        return model.get_memory_footprint()   # reports bitsandbytes / GPTQ / AWQ weights correctly
    return sum(tensor_bytes(p) for p in model.parameters()) + sum(tensor_bytes(b) for b in model.buffers())


def no_grad(precision: str):
    # quanto repacks INT4 weights on first use, which inference mode forbids.
    return torch.no_grad() if precision.startswith("quanto") else torch.inference_mode()


SANITY_TEXT = ("The quick brown fox jumps over the lazy dog. Paris is the capital of France, "
               "and Berlin is the capital of Germany. Water boils at one hundred degrees Celsius.")
SANITY_MAX_PPL = 200   # working instruct models score roughly 3-30 on this text; a broken load scores in the thousands


def tie_weights_if_needed(model, log):
    """Models with tie_word_embeddings share one matrix between the input embeddings and the output
    layer. Some transformers versions fail to tie them on load and leave the output layer randomly
    initialised, which turns every reply into random multilingual tokens."""
    if not getattr(model.config, "tie_word_embeddings", False):
        return
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    if head is None or emb is None or head.weight is emb.weight or head.weight.data_ptr() == emb.weight.data_ptr():
        return
    log("Output layer was not tied to the input embeddings after loading (a transformers loading bug); tying it")
    try:
        model.tie_weights()
    except Exception:  # noqa: BLE001
        pass
    if model.get_output_embeddings().weight.data_ptr() != emb.weight.data_ptr():
        model.get_output_embeddings().weight = emb.weight


def sanity_perplexity(model, tok, device, precision) -> float:
    ids = tok(SANITY_TEXT, return_tensors="pt").to(device)
    with no_grad(precision):
        loss = model(**ids, labels=ids["input_ids"]).loss.float().item()
    return float("nan") if loss != loss else float(min(torch.exp(torch.tensor(loss)).item(), 1e9))


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


class Slot:
    """One model under test (A = baseline, B = quantized)."""

    def __init__(self, name: str, repo: str, precision: str, device: str):
        self.name = name
        self.busy = threading.Lock()  # held while loading or generating
        self.defaults = {"repo": repo, "precision": precision, "device": device}
        self._clear()

    def _clear(self):
        self.state = "empty"  # empty | loading | ready | error
        self.stage = ""
        self.log = []
        self.repo = self.precision = self.device = None
        self.model = self.tok = None
        self.info = {}
        self.error = None
        self.warning = None
        self.blob_dir = None
        self.dl_total = 0
        self.t_start = time.time()

    def add_log(self, msg: str):
        self.log.append({"t": round(time.time() - self.t_start, 2), "msg": msg})
        print(f"[{self.name}] {msg}", flush=True)

    def snapshot(self):
        if self.stage == "download" and self.blob_dir:
            dl_done = min(dir_size(self.blob_dir), self.dl_total)
        else:
            dl_done = self.dl_total if self.state in ("ready", "loading") and self.stage != "download" else 0
        return {
            "name": self.name, "state": self.state, "stage": self.stage, "log": self.log[-30:],
            "repo": self.repo, "precision": self.precision, "device": self.device,
            "info": self.info, "error": self.error or self.warning, "warning": self.warning, "defaults": self.defaults,
            "download": {"done": dl_done, "total": self.dl_total},
            "gpu_alloc": device_alloc(self.device or ""),
            "elapsed": round(time.time() - self.t_start, 1) if self.state == "loading" else None,
        }

    # ---------------------------------------------------------------- loading
    def load(self, repo: str, precision: str, device: str):
        self._clear()
        self.repo, self.precision, self.device = repo, precision, device
        self.state = "loading"
        try:
            self.stage = "download"
            local = self._download(repo)

            self.stage = "tokenizer"
            self.add_log("Loading tokenizer")
            tok = AutoTokenizer.from_pretrained(local)

            self.stage = "weights"
            self.add_log(f"Loading weights to {device} as {PRECISIONS[precision]}")
            if precision == "quanto-int4" and device == "cpu":
                self.add_log("The first INT4 load on a CPU compiles quanto's kernels (a minute or two; needs a C++ compiler)")
            t = time.perf_counter()
            kwargs = dict(device_map={"": device}, low_cpu_mem_usage=True, **self._quant_kwargs(precision, device))
            try:
                # Explicit SDPA attention: with recent transformers the "eager" path overflows in FP16 and
                # Qwen2.5-family models then emit NaN logits ("!!!!" or multilingual garbage).
                model = AutoModelForCausalLM.from_pretrained(local, attn_implementation="sdpa", **kwargs).eval()
            except (ValueError, ImportError) as e:   # architecture without SDPA support
                self.add_log(f"SDPA attention unavailable ({e}); using the model's default")
                model = AutoModelForCausalLM.from_pretrained(local, **kwargs).eval()
            load_s = time.perf_counter() - t
            self.add_log(f"Weights loaded in {load_s:.1f}s")
            tie_weights_if_needed(model, self.add_log)
            ppl = sanity_perplexity(model, tok, device, precision)
            self.add_log(f"Sanity check: perplexity {ppl:.1f} on a plain English sentence" if ppl == ppl else "Sanity check: NaN outputs")
            if not ppl < SANITY_MAX_PPL:
                self.warning = (f"Loaded, but this model's outputs look broken on this machine (sanity perplexity "
                                f"{'NaN' if ppl != ppl else f'{ppl:.0f}'}; a working model scores under {SANITY_MAX_PPL}). "
                                "Expect garbage text. Try another precision, or restart the backend.")
                self.add_log("WARNING: " + self.warning)

            self.stage = "warmup"
            self.add_log("Warm-up generation (compiles kernels so the first real run is fair)")
            ids = tok("def hello_world():", return_tensors="pt").to(device)
            with no_grad(precision):
                model.generate(**ids, max_new_tokens=8, do_sample=False,
                               pad_token_id=pad_id(tok))

            self.model, self.tok = model, tok
            self.info = {
                "load_s": round(load_s, 2),
                "total_s": round(time.time() - self.t_start, 2),
                "footprint": footprint(model, precision),
                "gpu_alloc": device_alloc(device),
                "disk": self.dl_total,
                "quant": self._describe_quant(model, precision),
                "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
                "attn": getattr(model.config, "_attn_implementation", None), "sanity_ppl": round(ppl, 2) if ppl == ppl else None,
                "versions": VERSIONS,
            }
            self.stage = "ready"
            self.state = "ready"
            self.add_log("Ready")
        except Exception as e:  # noqa: BLE001 - surface any load failure in the UI
            traceback.print_exc()
            self.model = self.tok = None
            gc.collect()
            empty_cache()
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"
            if precision.startswith("quanto") and any(w in str(e) for w in ("Ninja", "building extension", "compiler", "cl.exe")):
                self.error += (" · Building quanto's kernels needs a C++ compiler (macOS: xcode-select --install; "
                               "Windows: Visual Studio Build Tools). INT8 usually works without one.")
            self.add_log(f"Failed: {self.error}")

    def _download(self, repo: str) -> str:
        if os.path.isdir(repo):
            self.add_log(f"Using local directory {repo}")
            return repo
        self.add_log(f"Resolving {repo} on the Hugging Face Hub")
        siblings = HfApi().model_info(repo, files_metadata=True).siblings
        names = [s.rfilename for s in siblings]
        weights = ["*.safetensors"] if any(n.endswith(".safetensors") for n in names) else ["*.bin"]
        patterns = META_PATTERNS + weights
        wanted = [s for s in siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
        self.dl_total = sum(s.size or 0 for s in wanted)
        cap = LIMITS["max_download_bytes"]
        if cap and self.dl_total > cap:
            raise ValueError(f"{repo} needs {self.dl_total / 1e9:.1f} GB, above this server's {cap / 1e9:.0f} GB download limit")
        self.blob_dir = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "blobs"
        self.add_log(f"Downloading {len(wanted)} files ({self.dl_total / 1e9:.2f} GB, cached files are reused)")
        t = time.perf_counter()
        with DOWNLOAD_LOCK:
            local = snapshot_download(repo, allow_patterns=patterns)
        self.add_log(f"Files ready in {time.perf_counter() - t:.1f}s")
        return local

    @staticmethod
    def _quant_kwargs(precision: str, device: str) -> dict:
        problem = precision_problems().get(precision)
        if problem:
            raise ValueError(f"{PRECISIONS[precision]} {problem} (this machine: {accelerator()})")
        if precision.startswith("bnb") and not device.startswith("cuda"):
            raise ValueError("bitsandbytes quantization runs on a cuda device only")
        if precision.startswith("quanto"):
            from transformers import QuantoConfig

            # CPUs compute fastest (and most reliably) in FP32; GPUs in FP16.
            dtype = torch.float32 if device == "cpu" else torch.float16
            return {"quantization_config": QuantoConfig(weights=precision.split("-")[1]), "torch_dtype": dtype}
        if precision == "fp16":
            return {"torch_dtype": torch.float16}
        if precision == "bf16":
            return {"torch_dtype": torch.bfloat16}
        if precision == "fp32":
            return {"torch_dtype": torch.float32}
        if precision == "bnb-8bit":
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "torch_dtype": torch.float16}
        if precision == "bnb-4bit":
            return {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            ), "torch_dtype": torch.float16}
        if precision == "prequantized":
            return {"torch_dtype": "auto"}
        raise ValueError(f"Unknown precision {precision}")

    @staticmethod
    def _describe_quant(model, precision: str) -> str:
        qc = getattr(model.config, "quantization_config", None)
        if precision == "prequantized" and qc is not None:
            qc = qc if isinstance(qc, dict) else qc.to_dict()
            bits = qc.get("bits") or qc.get("w_bit")
            return f"{qc.get('quant_method', 'quantized')} {bits}-bit" if bits else str(qc.get("quant_method"))
        return PRECISIONS[precision]

    def unload(self):
        self.model = self.tok = None
        gc.collect()
        empty_cache()
        self._clear()


# -------------------------------------------------------------------- generation
STOP = threading.Event()


class StopOnFlag(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), STOP.is_set(), dtype=torch.bool, device=input_ids.device)


class CountingStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that also counts generated tokens and records time to first token."""

    def __init__(self, tok):
        super().__init__(tok, skip_prompt=True, skip_special_tokens=True)
        self.n_tokens = 0
        self.first_token_time = None

    def put(self, value):
        if not (self.skip_prompt and self.next_tokens_are_prompt):
            if self.first_token_time is None:
                self.first_token_time = time.perf_counter()
            self.n_tokens += value.numel()
        super().put(value)


def pad_id(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def build_prompt(tok, system: str, prompt: str) -> str:
    if tok.chat_template:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{prompt}\n" if system else prompt  # base (non-chat) model


def run_generation(slot: Slot, req: "GenRequest", events: queue.Queue):
    if not slot.busy.acquire(blocking=False):
        events.put({"slot": slot.name, "type": "error", "msg": "Model is busy"})
        return
    try:
        tok, model, device = slot.tok, slot.model, slot.device
        events.put({"slot": slot.name, "type": "start"})
        inputs = tok(build_prompt(tok, req.system, req.prompt), return_tensors="pt").to(device)
        n_prompt = inputs["input_ids"].shape[1]
        streamer = CountingStreamer(tok)
        kwargs = dict(**inputs, max_new_tokens=req.max_new_tokens, streamer=streamer,
                      pad_token_id=pad_id(tok),
                      stopping_criteria=StoppingCriteriaList([StopOnFlag()]))
        if req.temperature > 0:
            kwargs.update(do_sample=True, temperature=req.temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        idx = cuda_index(device)
        if idx is not None:
            torch.cuda.synchronize(idx)
            torch.cuda.reset_peak_memory_stats(idx)
        result, failure = {}, []

        def _generate():
            try:
                with no_grad(slot.precision):
                    result["out"] = model.generate(**kwargs)
            except Exception as e:  # noqa: BLE001
                failure.append(e)
                streamer.end()

        t0 = time.perf_counter()
        worker = threading.Thread(target=_generate, daemon=True)
        worker.start()
        for chunk in streamer:
            events.put({"slot": slot.name, "type": "token", "text": chunk,
                        "n": streamer.n_tokens, "t": round(time.perf_counter() - t0, 4)})
        worker.join()
        t_end = time.perf_counter()
        if failure:
            raise failure[0]

        n = streamer.n_tokens
        first = streamer.first_token_time or t_end
        text = tok.decode(result["out"][0][n_prompt:], skip_special_tokens=True)
        events.put({"slot": slot.name, "type": "done", "text": text, "metrics": {
            "prompt_tokens": n_prompt,
            "new_tokens": n,
            "ttft_s": round(first - t0, 4),
            "total_s": round(t_end - t0, 3),
            "decode_tps": round((n - 1) / (t_end - first), 2) if n > 1 and t_end > first else None,
            "e2e_tps": round(n / (t_end - t0), 2) if n else None,
            "peak_mem": torch.cuda.max_memory_allocated(idx) if idx is not None else None,
            "stopped": STOP.is_set(),
        }})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        events.put({"slot": slot.name, "type": "error", "msg": f"{type(e).__name__}: {e}"})
    finally:
        slot.busy.release()


# ------------------------------------------------------------------------- API
class AgentRequest(BaseModel):
    benchmark: str = "bfcl_multi"
    index: int
    slots: list[str] = ["a", "b"]
    mode: str = "sequential"   # sequential | parallel
    max_steps: int = agent_bench.MAX_STEPS
    max_new_tokens: int = 512
    temperature: float = 0.0
    loop_guard: int = agent_runner.LOOP_GUARD


class LoadRequest(BaseModel):
    slot: str
    repo: str
    precision: str
    device: str


class GenRequest(BaseModel):
    prompt: str
    system: str = ""
    slots: list[str] = ["a", "b"]
    max_new_tokens: int = 512
    temperature: float = 0.0
    mode: str = "sequential"  # sequential | parallel (parallel shares the GIL, so tok/s drops)


class GradeRequest(BaseModel):
    text: str
    tests: list[str] = []          # built-in tasks: assert statements
    benchmark: str | None = None   # or a benchmark problem graded with its official tests
    index: int | None = None


def grading_payload(req: GradeRequest) -> dict:
    if req.benchmark is not None:
        if req.benchmark not in benchmarks.BENCHMARKS or req.index is None:
            raise HTTPException(400, "Unknown benchmark problem")
        return benchmarks.build_grading(req.benchmark, req.index, req.text)
    tests = [t for t in (x.strip() for x in req.tests) if t]
    return {"code": benchmarks.extract_code(req.text), "tests": tests, "kind": "asserts"}


def execute_tests(payload: dict, timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "payload.json").write_text(json.dumps(payload))
        # Run inside a throwaway dir with its own HOME so file-writing tasks stay contained; no GPUs.
        env = {**os.environ, "HOME": d, "TMPDIR": d, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": ""}
        try:
            proc = subprocess.run([sys.executable, "-I", str(TEST_RUNNER), "payload.json"],
                                  capture_output=True, text=True, timeout=timeout, cwd=d, env=env)
        except subprocess.TimeoutExpired:
            return {"compiled": True, "error": f"Timed out after {timeout}s", "results": [], "timeout": True}
    line = next((l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")), None)
    if line is None:
        return {"compiled": False, "error": (proc.stderr.strip() or "Process exited without a result")[-500:], "results": []}
    return json.loads(line[len("__RESULT__"):])


def create_app(args) -> FastAPI:
    app = FastAPI(title="QuantBench Live")
    store = RunStore(Path(args.db))
    agent_store = AgentStore(Path(args.db))
    n_gpu = torch.cuda.device_count()
    dev_a = "cuda:0" if n_gpu else "mps" if HAS_MPS else "cpu"
    dev_b = "cuda:1" if n_gpu > 1 else dev_a
    model = args.model or ("Qwen/Qwen2.5-Coder-1.5B-Instruct" if HAS_CUDA else SMALL_MODEL)
    quant = args.quant or ("bnb-4bit" if HAS_CUDA else "quanto-int8")
    slots = {
        "a": Slot("a", model, "fp32" if dev_a == "cpu" else "fp16", dev_a),
        "b": Slot("b", args.quant_model or model, quant, dev_b),
    }

    if args.auth_key:
        @app.middleware("http")
        async def require_key(request: Request, call_next):
            # Every API call except the health check must carry this backend's key.
            open_path = request.url.path in ("/api/health",) or not request.url.path.startswith("/api/")
            if request.method != "OPTIONS" and not open_path:
                given = request.headers.get("x-qb-key") or ""
                if not secrets.compare_digest(given, args.auth_key):
                    return JSONResponse({"detail": "Missing or wrong backend key"}, status_code=401)
            return await call_next(request)

    # Added after the key check so it wraps it: rejected requests still carry CORS headers.
    app.add_middleware(CORSMiddleware, allow_origins=args.cors_origins.split(","), allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def private_network_access(request: Request, call_next):
        # Chrome asks before a public https site may call http://127.0.0.1 (the "own computer" setup); say yes.
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network"):
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    def get_slot(name: str) -> Slot:
        if name not in slots:
            raise HTTPException(404, f"Unknown slot {name}")
        return slots[name]

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    site_config = Path(args.site_config).read_text() if args.site_config else '{"mode": "local"}'

    @app.get("/config.js")
    def config_js():
        # Standalone: no login, history in SQLite. With --site-config (connect.py --local) this server
        # serves the hosted site's page itself, signed in to the shared history, with no cross-site calls.
        return Response(f"window.QB_CONFIG = {site_config};", media_type="application/javascript")

    @app.get("/api/health")
    def health():
        return {"ok": True, "app": "quantbench", "version": 2, "gpus": gpu_list(), "accelerator": accelerator(), "versions": VERSIONS,
                "platform": platform.system(), "key_required": bool(args.auth_key)}

    @app.get("/api/config")
    def config():
        devices = [f"cuda:{i}" for i in range(n_gpu)] + (["mps"] if HAS_MPS else []) + ["cpu"]
        return {"presets": PRESETS, "precisions": PRECISIONS, "precision_problems": precision_problems(),
                "devices": devices, "models": model_catalog(),
                "benchmarks": {k: {"name": v["name"], "count": v["count"], "kind": v["kind"]}
                               for k, v in benchmarks.BENCHMARKS.items()}}

    @app.get("/api/status")
    def status():
        return {"slots": {k: s.snapshot() for k, s in slots.items()}, "gpus": gpu_list()}

    @app.post("/api/load")
    def load(req: LoadRequest):
        slot = get_slot(req.slot)
        if req.precision not in PRECISIONS:
            raise HTTPException(400, f"Unknown precision {req.precision}")
        if not slot.busy.acquire(blocking=False):
            raise HTTPException(409, "Slot is busy (loading or generating)")

        def _job():
            try:
                if slot.model is not None:
                    slot.unload()
                slot.load(req.repo.strip(), req.precision, req.device)
            finally:
                slot.busy.release()

        threading.Thread(target=_job, daemon=True).start()
        return {"ok": True}

    @app.post("/api/unload/{name}")
    def unload(name: str):
        slot = get_slot(name)
        if not slot.busy.acquire(blocking=False):
            raise HTTPException(409, "Slot is busy (loading or generating)")
        try:
            slot.unload()
        finally:
            slot.busy.release()
        return {"ok": True}

    @app.post("/api/stop")
    def stop():
        STOP.set()
        return {"ok": True}

    @app.post("/api/generate")
    def generate(req: GenRequest):
        targets = [slots[s] for s in req.slots if s in slots and slots[s].state == "ready"]
        if not targets:
            raise HTTPException(409, "No model is loaded")
        STOP.clear()
        events: queue.Queue = queue.Queue()

        def _sequential():
            for s in targets:
                run_generation(s, req, events)

        if req.mode == "sequential":
            threads = [threading.Thread(target=_sequential, daemon=True)]
        else:
            threads = [threading.Thread(target=run_generation, args=(s, req, events), daemon=True) for s in targets]
        for t in threads:
            t.start()

        def stream():
            yield "data: " + json.dumps({"type": "begin", "slots": [s.name for s in targets], "mode": req.mode}) + "\n\n"
            remaining, texts = len(targets), {}
            while remaining:
                ev = events.get()
                if ev["type"] in ("done", "error"):
                    remaining -= 1
                    if ev["type"] == "done":
                        texts[ev["slot"]] = ev["text"]
                yield "data: " + json.dumps(ev) + "\n\n"
            if len(texts) == 2:
                a, b = texts.values()
                sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
                yield "data: " + json.dumps({"type": "compare", "similarity": round(sim, 4), "identical": a == b}) + "\n\n"
            yield "data: " + json.dumps({"type": "end"}) + "\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/benchmarks/{name}")
    def benchmark_list(name: str):
        if name not in benchmarks.BENCHMARKS:
            raise HTTPException(404, "Unknown benchmark")
        try:
            return {"problems": benchmarks.listing(name)}
        except Exception as e:  # noqa: BLE001 - e.g. no network access to the Hub
            raise HTTPException(502, f"Could not load {name}: {type(e).__name__}: {e}") from e

    @app.get("/api/benchmarks/{name}/{index}")
    def benchmark_problem(name: str, index: int):
        if name not in benchmarks.BENCHMARKS or not 0 <= index < len(benchmarks.load(name)):
            raise HTTPException(404, "Unknown benchmark problem")
        return benchmarks.problem(name, index)

    @app.post("/api/run_tests")
    def run_tests(req: GradeRequest):
        payload = grading_payload(req)
        timeout = args.test_timeout if payload["kind"] == "asserts" else args.bench_timeout
        res = execute_tests(payload, timeout)
        res["total"] = len(payload["tests"]) if payload["kind"] == "asserts" else len(res["results"])
        res["passed"] = sum(r["ok"] for r in res["results"])
        # pass@1 for this sample: code ran, at least one test ran, and every test passed
        res["all_passed"] = bool(res["compiled"] and not res.get("error") and res["results"]
                                 and res["passed"] == len(res["results"]) and res["passed"] >= res["total"])
        res["code"] = payload["code"]
        return res

    @app.post("/api/analyze")
    def analyze(req: GradeRequest):
        payload = grading_payload(req)
        return analysis.analyze(payload.get("analysis_code", payload["code"]))

    @app.post("/api/summarize")
    def summarize_run(record: dict):
        return summarize(record)

    # Local history (SQLite). The hosted site stores runs in Supabase instead.
    @app.get("/api/history/rows")
    def history_rows():
        return {"rows": store.rows()}

    @app.delete("/api/history/{key}")
    def delete_history_group(key: str):
        if not store.delete_group(key):
            raise HTTPException(404, "Run not found")
        return {"ok": True}

    @app.post("/api/runs")
    def save_run(record: dict):
        return store.add(record)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return run

    # ------------------------------------------------------------ agent page (tool-calling trajectories)
    @app.get("/agent")
    def agent_page():
        return FileResponse(ROOT / "static" / "agent.html")

    def get_bench(key: str):
        if key not in agent_benchmarks.BENCHMARKS:
            raise HTTPException(404, f"Unknown benchmark {key}")
        return agent_benchmarks.BENCHMARKS[key]

    @app.get("/api/agent/config")
    def agent_config():
        return {"benchmarks": [b.info() for b in agent_benchmarks.BENCHMARKS.values()], "loop_guard": agent_runner.LOOP_GUARD,
                "models": agent_catalog(), "precisions": PRECISIONS, "precision_problems": precision_problems(),
                "devices": [f"cuda:{i}" for i in range(n_gpu)] + (["mps"] if HAS_MPS else []) + ["cpu"]}

    @app.get("/api/agent/benchmarks/{key}")
    def agent_listing(key: str):
        bench = get_bench(key)
        try:
            return {**bench.info(), "tasks": bench.listing()}
        except Exception as e:  # noqa: BLE001 - e.g. the Hub is unreachable for ToolHop's data
            raise HTTPException(503, f"Could not load {bench.name}: {type(e).__name__}: {e}") from e

    @app.get("/api/agent/tasks/{key}/{index}")
    def agent_task(key: str, index: int):
        bench = get_bench(key)
        if not 0 <= index < len(bench.listing()):
            raise HTTPException(404, "No such task")
        return bench.detail(index)

    @app.post("/api/agent/run")
    def agent_run(req: AgentRequest):
        bench = get_bench(req.benchmark)
        if not 0 <= req.index < len(bench.listing()):
            raise HTTPException(404, "No such task")
        targets = [slots[s] for s in req.slots if s in slots and slots[s].state == "ready"]
        if not targets:
            raise HTTPException(409, "No model is loaded")
        task = bench.task(req.index)
        opts = {"max_steps": max(1, min(req.max_steps, 50)), "max_new_tokens": max(16, min(req.max_new_tokens, 4096)),
                "temperature": req.temperature, "loop_guard": max(0, min(req.loop_guard, 20))}
        STOP.clear()
        events: queue.Queue = queue.Queue()

        def run_one(slot: Slot):
            if not slot.busy.acquire(blocking=False):
                events.put({"slot": slot.name, "type": "error", "msg": "Model is busy"})
                return
            try:
                events.put({"slot": slot.name, "type": "start", "native_tools": agent_runner.native_tools(slot.tok)})
                ep = agent_runner.run_episode(slot, bench, task, opts, lambda e: events.put({"slot": slot.name, **e}),
                                              STOP, no_grad(slot.precision))
                events.put({"slot": slot.name, "type": "done", "episode": ep})
            except Exception as e:  # noqa: BLE001 - shown in the UI
                traceback.print_exc()
                events.put({"slot": slot.name, "type": "error", "msg": f"{type(e).__name__}: {e}"})
            finally:
                slot.busy.release()

        if req.mode == "sequential":
            threads = [threading.Thread(target=lambda: [run_one(s) for s in targets], daemon=True)]
        else:
            threads = [threading.Thread(target=run_one, args=(s,), daemon=True) for s in targets]
        for t in threads:
            t.start()

        def stream():
            yield "data: " + json.dumps({"type": "begin", "slots": [s.name for s in targets], "mode": req.mode, "opts": opts}) + "\n\n"
            remaining = len(targets)
            while remaining:
                ev = events.get()
                if ev["type"] in ("done", "error"):
                    remaining -= 1
                yield "data: " + json.dumps(ev, default=str) + "\n\n"
            yield "data: " + json.dumps({"type": "end"}) + "\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/agent/summarize")
    def agent_summarize(record: dict):
        return summarize_agent(record)

    @app.get("/api/agent/history/rows")
    def agent_history_rows():
        return {"rows": agent_store.rows()}

    @app.delete("/api/agent/history/{key}")
    def agent_history_delete(key: str):
        if not agent_store.delete_group(key):
            raise HTTPException(404, "Run not found")
        return {"ok": True}

    @app.post("/api/agent/runs")
    def agent_save(record: dict):
        return agent_store.add(record)

    @app.get("/api/agent/runs/{run_id}")
    def agent_get(run_id: int):
        run = agent_store.get(run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return run

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=None,
                   help="Baseline model (slot A default; Qwen2.5-Coder-1.5B-Instruct with an NVIDIA GPU, 0.5B otherwise)")
    p.add_argument("--quant-model", default=None, help="Slot B repo default (defaults to --model)")
    p.add_argument("--quant", default=None, choices=list(PRECISIONS),
                   help="Slot B precision default (bnb-4bit with an NVIDIA GPU, quanto-int8 otherwise)")
    p.add_argument("--test-timeout", type=int, default=10, help="Seconds allowed for built-in task / HumanEval tests")
    p.add_argument("--db", default=str(ROOT / "runs" / "history.sqlite"), help="Where run history is stored")
    p.add_argument("--auth-key", default=os.environ.get("QB_AUTH_KEY"),
                   help="Require this key (X-QB-Key header) on every API call; set when the server is reachable from outside")
    p.add_argument("--cors-origins", default="*", help="Comma-separated origins allowed to call the API from a browser")
    p.add_argument("--max-download-gb", type=float, default=0, help="Refuse models whose files exceed this size (0 = no limit)")
    p.add_argument("--bench-timeout", type=int, default=60, help="Seconds allowed for a BigCodeBench test suite")
    p.add_argument("--site-config", default=None, help="JSON file with the hosted site's config (set by connect.py --local)")
    args = p.parse_args()
    LIMITS["max_download_bytes"] = args.max_download_gb * 1e9 or None

    import uvicorn

    app = create_app(args)
    # Printing the URL lets VS Code Remote-SSH detect the port and offer "Open in Browser".
    print(f"QuantBench Live running at http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
