from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .infer_utils import find_repo_root, load_env


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(os.path.expanduser(raw_path))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    return value


@dataclass(frozen=True)
class RuntimePaths:
    repo_root: Path
    datasets_dir: Path
    models_dir: Path
    meddinov3_ckpt_path: Path


@dataclass(frozen=True)
class RuntimeSources:
    meddinov3_gdrive_file_id: str
    meddinov3_url: str
    meddinov3_hf_repo: str
    dino_url: str
    hf_mirror: str


@dataclass(frozen=True)
class InferDefaults:
    output_dir: str = "outputs/meddinov3"
    stride: int = 1
    max_slices: int = 0
    batch_size: int = 8
    resize: int = 224
    sim_slice: int = -1
    sim_patch: str = "center"
    device: str = ""


@dataclass(frozen=True)
class TrainDefaults:
    output_dir: str = "checkpoints/head_train"
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    hidden_dim: int = 256
    dropout: float = 0.2
    seed: int = 42
    device: str = ""


@dataclass(frozen=True)
class EvalDefaults:
    batch_size: int = 256
    device: str = ""


INFER_DEFAULTS = InferDefaults()
TRAIN_DEFAULTS = TrainDefaults()
EVAL_DEFAULTS = EvalDefaults()


def get_runtime_paths() -> RuntimePaths:
    load_env()
    repo_root = find_repo_root()
    models_dir = _resolve_repo_path(repo_root, _env_str("MODELS_DIR", "modelsweights"))
    datasets_dir = _resolve_repo_path(repo_root, _env_str("DATASETS_DIR", "datasets_local"))
    med_ckpt_default = "modelsweights/meddinov3/MedDINOv3-ViTB-16-CT-3M/model.pth"
    med_ckpt_path = _resolve_repo_path(repo_root, _env_str("MEDDINOV3_CKPT_PATH", med_ckpt_default))
    return RuntimePaths(
        repo_root=repo_root,
        datasets_dir=datasets_dir,
        models_dir=models_dir,
        meddinov3_ckpt_path=med_ckpt_path,
    )


def get_runtime_sources() -> RuntimeSources:
    load_env()
    return RuntimeSources(
        meddinov3_gdrive_file_id=_env_str("MEDDINOV3_GDRIVE_FILE_ID"),
        meddinov3_url=_env_str("MEDDINOV3_URL"),
        meddinov3_hf_repo=_env_str("MEDDINOV3_HF_REPO"),
        dino_url=_env_str("DINOV3_URL"),
        hf_mirror=_env_str("HF_MIRROR"),
    )
