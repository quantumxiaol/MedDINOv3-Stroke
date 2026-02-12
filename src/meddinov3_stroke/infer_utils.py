from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".env").exists():
            return parent
    return Path.cwd()


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = find_repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def get_env_path(key: str) -> Path:
    value = os.getenv(key)
    if not value:
        raise FileNotFoundError(f"{key} is not set in environment or .env.")
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = (find_repo_root() / path).resolve()
    return path


def find_input_nii() -> Path:
    root = find_repo_root()
    input_dir = root / "input"
    if not input_dir.exists():
        raise FileNotFoundError("input/ directory not found.")
    for pattern in ("*.nii.gz", "*.nii"):
        hits = sorted(input_dir.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError("No .nii or .nii.gz file found under input/.")


def _cuda_available(torch_module: Any) -> bool:
    try:
        return torch_module.cuda.is_available() and torch_module.cuda.device_count() > 0
    except Exception:
        return False


def _parse_device_override(torch_module: Any, value: str):
    val = value.strip().lower()
    if not val:
        return None
    if val in ("cpu", "mps"):
        return torch_module.device(val), val
    if val in ("cuda", "gpu"):
        if _cuda_available(torch_module):
            return torch_module.device("cuda:0"), "gpu"
        return None
    if val.isdigit():
        if _cuda_available(torch_module):
            return torch_module.device(f"cuda:{val}"), f"gpu:{val}"
        return None
    if val.startswith("gpu:"):
        idx = val.split(":", 1)[1]
        if _cuda_available(torch_module):
            return torch_module.device(f"cuda:{idx}"), f"gpu:{idx}"
        return None
    if val.startswith("cuda:"):
        idx = val.split(":", 1)[1]
        if _cuda_available(torch_module):
            return torch_module.device(val), f"gpu:{idx}"
        return None
    return None


def select_device(torch_module: Any):
    override = os.getenv("CT_MODEL_DEVICE")
    if override:
        parsed = _parse_device_override(torch_module, override)
        if parsed is not None:
            return parsed
    if _cuda_available(torch_module):
        return torch_module.device("cuda:0"), "gpu"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps"), "mps"
    return torch_module.device("cpu"), "cpu"


def _sync_device(torch_module: Any, device: Any):
    if device.type == "cuda":
        torch_module.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch_module, "mps"):
        try:
            torch_module.mps.synchronize()
        except Exception:
            pass


def _gpu_mem_mb(torch_module: Any, device: Any):
    if device.type == "cuda":
        return torch_module.cuda.max_memory_allocated(device) / (1024**2)
    if device.type == "mps" and hasattr(torch_module, "mps"):
        try:
            return torch_module.mps.current_allocated_memory() / (1024**2)
        except Exception:
            return None
    return None


def _cpu_rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > 10**8:  # bytes (macOS)
                return rss / (1024**2)
            return rss / 1024  # KB (Linux)
        except Exception:
            return None


def measure_inference(torch_module: Any, device: Any, fn):
    if device.type == "cuda" and _cuda_available(torch_module):
        try:
            torch_module.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    _sync_device(torch_module, device)
    start = time.perf_counter()
    result = fn()
    _sync_device(torch_module, device)
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "gpu_mem_mb": _gpu_mem_mb(torch_module, device),
        "cpu_rss_mb": _cpu_rss_mb(),
        "result": result,
    }


def print_stats(label: str, stats: dict):
    gpu_mem = stats.get("gpu_mem_mb")
    cpu_rss = stats.get("cpu_rss_mb")
    elapsed = stats.get("elapsed_s")
    print(f"[{label}] elapsed_s={elapsed:.3f}")
    if gpu_mem is not None:
        print(f"[{label}] gpu_mem_mb={gpu_mem:.1f}")
    if cpu_rss is not None:
        print(f"[{label}] cpu_rss_mb={cpu_rss:.1f}")

