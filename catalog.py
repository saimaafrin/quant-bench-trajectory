"""Models offered in the UI dropdown (all open / non-gated on the Hub, <= ~7B parameters)."""
import importlib.util

# params in billions, used for the memory estimate shown before loading
MODELS = [
    {"group": "Qwen2.5-Coder (instruct)", "repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "params": 0.49},
    {"group": "Qwen2.5-Coder (instruct)", "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "params": 1.54},
    {"group": "Qwen2.5-Coder (instruct)", "repo": "Qwen/Qwen2.5-Coder-3B-Instruct", "params": 3.09},
    {"group": "Qwen2.5-Coder (instruct)", "repo": "Qwen/Qwen2.5-Coder-7B-Instruct", "params": 7.62},
    {"group": "DeepSeek-Coder (instruct)", "repo": "deepseek-ai/deepseek-coder-1.3b-instruct", "params": 1.35},
    {"group": "DeepSeek-Coder (instruct)", "repo": "deepseek-ai/deepseek-coder-6.7b-instruct", "params": 6.74},
    {"group": "Other instruct models", "repo": "codellama/CodeLlama-7b-Instruct-hf", "params": 6.74},
    {"group": "Other instruct models", "repo": "ibm-granite/granite-3b-code-instruct-128k", "params": 3.48},
    {"group": "Other instruct models", "repo": "microsoft/Phi-3.5-mini-instruct", "params": 3.82},
    {"group": "Base models (completion only, no chat template)", "repo": "bigcode/starcoder2-3b", "params": 3.03},
    {"group": "Base models (completion only, no chat template)", "repo": "bigcode/starcoder2-7b", "params": 7.17},
    {"group": "Pre-quantized (use precision: Pre-quantized)", "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ", "params": 1.54, "needs": "awq"},
    {"group": "Pre-quantized (use precision: Pre-quantized)", "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4", "params": 1.54, "needs": "gptq"},
    {"group": "Pre-quantized (use precision: Pre-quantized)", "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", "params": 7.62, "needs": "awq"},
    {"group": "Pre-quantized (use precision: Pre-quantized)", "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4", "params": 7.62, "needs": "gptq"},
    {"group": "Pre-quantized (use precision: Pre-quantized)", "repo": "TheBloke/deepseek-coder-6.7B-instruct-AWQ", "params": 6.74, "needs": "awq"},
    {"group": "Pre-quantized (use precision: Pre-quantized)", "repo": "TheBloke/deepseek-coder-6.7B-instruct-GPTQ", "params": 6.74, "needs": "gptq"},
]

# Python packages transformers needs to load each pre-quantized format.
BACKENDS = {
    "awq": {"label": "autoawq", "available": lambda: importlib.util.find_spec("awq") is not None},
    "gptq": {"label": "optimum + auto-gptq (or gptqmodel)",
             "available": lambda: importlib.util.find_spec("optimum") is not None and (
                 importlib.util.find_spec("auto_gptq") is not None or importlib.util.find_spec("gptqmodel") is not None)},
}


def model_catalog():
    out = []
    for m in MODELS:
        m = dict(m)
        need = m.pop("needs", None)
        if need:
            m["prequantized"] = True
            m["available"] = BACKENDS[need]["available"]()
            m["requires"] = BACKENDS[need]["label"]
        else:
            m["prequantized"] = False
            m["available"] = True
        out.append(m)
    return out


# Agent page: models whose chat template supports tool calling (others still work through a
# prompt-based fallback, which the trajectory records).
AGENT_MODELS = [
    {"group": "Qwen2.5 (instruct, native tool calling)", "repo": "Qwen/Qwen2.5-0.5B-Instruct", "params": 0.49},
    {"group": "Qwen2.5 (instruct, native tool calling)", "repo": "Qwen/Qwen2.5-1.5B-Instruct", "params": 1.54},
    {"group": "Qwen2.5 (instruct, native tool calling)", "repo": "Qwen/Qwen2.5-3B-Instruct", "params": 3.09},
    {"group": "Qwen2.5 (instruct, native tool calling)", "repo": "Qwen/Qwen2.5-7B-Instruct", "params": 7.62},
    {"group": "Trained for function calling (BFCL)", "repo": "MadeAgents/Hammer2.1-1.5b", "params": 1.54},
    {"group": "Trained for function calling (BFCL)", "repo": "MadeAgents/Hammer2.1-3b", "params": 3.09},
    {"group": "Trained for function calling (BFCL)", "repo": "MadeAgents/Hammer2.1-7b", "params": 7.61},
    {"group": "Trained for function calling (BFCL)", "repo": "Salesforce/xLAM-2-1b-fc-r", "params": 1.54},
    {"group": "Trained for function calling (BFCL)", "repo": "Salesforce/xLAM-2-3b-fc-r", "params": 3.09},
    {"group": "Qwen3 (needs transformers 4.51+)", "repo": "Qwen/Qwen3-1.7B", "params": 2.03, "needs": "qwen3"},
    {"group": "Qwen3 (needs transformers 4.51+)", "repo": "Qwen/Qwen3-4B", "params": 4.02, "needs": "qwen3"},
    {"group": "Qwen3 (needs transformers 4.51+)", "repo": "Qwen/Qwen3-8B", "params": 8.19, "needs": "qwen3"},
    {"group": "Qwen2.5-Coder (instruct)", "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "params": 1.54},
    {"group": "Qwen2.5-Coder (instruct)", "repo": "Qwen/Qwen2.5-Coder-7B-Instruct", "params": 7.62},
]


def _transformers_at_least(version: str) -> bool:
    try:
        import transformers
        from packaging.version import Version
        return Version(transformers.__version__) >= Version(version)
    except Exception:  # noqa: BLE001
        return False


AGENT_BACKENDS = {"qwen3": {"label": "transformers 4.51+", "available": lambda: _transformers_at_least("4.51")}}


def agent_catalog():
    out = []
    for m in AGENT_MODELS:
        m = dict(m)
        need = m.pop("needs", None)
        m["prequantized"] = False
        m["available"] = AGENT_BACKENDS[need]["available"]() if need else True
        if need:
            m["requires"] = AGENT_BACKENDS[need]["label"]
        out.append(m)
    return out
