from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import get_runtime_paths
from .infer_utils import measure_inference, select_device

LABEL_NAMES = (
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
)


@dataclass(frozen=True)
class SliceExtractConfig:
    stride: int = 1
    max_slices: int = 0
    batch_size: int = 8
    resize: int = 224


def aggregate_embeddings(slice_embeddings: np.ndarray, mode: str = "mean") -> np.ndarray:
    if slice_embeddings.ndim != 2:
        raise ValueError(f"Expected slice embeddings [N, D], got {slice_embeddings.shape}")
    if mode == "mean":
        return slice_embeddings.mean(axis=0)
    if mode == "meanmax":
        return np.concatenate([slice_embeddings.mean(axis=0), slice_embeddings.max(axis=0)], axis=0)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def infer_pool_mode(input_dim: int, slice_dim: int) -> str:
    if input_dim == slice_dim:
        return "mean"
    if input_dim == 2 * slice_dim:
        return "meanmax"
    raise ValueError(f"Cannot infer pool mode from input_dim={input_dim}, slice_dim={slice_dim}")


def load_volume_zyx(input_path: str | Path) -> np.ndarray:
    import nibabel as nib

    path = Path(input_path)
    img = nib.load(str(path)).get_fdata()
    if img.ndim == 4:
        img = img[..., 0]
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected volume shape: {arr.shape}")
    return np.transpose(arr, (2, 0, 1))  # D, H, W


class MedDINOv3FeatureExtractor:
    def __init__(self, device_override: str = ""):
        paths = get_runtime_paths()
        ckpt_path = paths.meddinov3_ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MedDINOv3 weights not found: {ckpt_path}")
        if device_override:
            os.environ["CT_MODEL_DEVICE"] = device_override
        self.repo_root = paths.repo_root
        self.torch = self._lazy_import_torch_and_prepare_path()
        self.device, self.device_name = select_device(self.torch)
        self.model = self._load_model(ckpt_path)
        self.mean = 65.1084213256836
        self.std = 178.01663208007812
        patch_size = getattr(self.model, "patch_size", 16)
        self.patch_size = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)

    def _lazy_import_torch_and_prepare_path(self):
        dinov3_parent = (
            self.repo_root
            / "third_party"
            / "MedDINOv3"
            / "nnUNet"
            / "nnunetv2"
            / "training"
            / "nnUNetTrainer"
            / "dinov3"
        )
        if str(dinov3_parent) not in sys.path:
            sys.path.insert(0, str(dinov3_parent))
        import torch

        return torch

    def _load_model(self, ckpt_path: Path):
        from dinov3.models.vision_transformer import vit_base

        model = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1.0e-05,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )
        chkpt = self.torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
        state = chkpt.get("teacher", chkpt)
        if not isinstance(state, dict):
            raise ValueError("Unexpected MedDINOv3 checkpoint format.")
        state = {
            k.replace("backbone.", ""): v
            for k, v in state.items()
            if "ibot" not in k and "dino_head" not in k
        }
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _validate_extract_config(self, cfg: SliceExtractConfig) -> None:
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if cfg.resize <= 0:
            raise ValueError("resize must be > 0")
        if cfg.stride <= 0:
            raise ValueError("stride must be > 0")
        if cfg.max_slices < 0:
            raise ValueError("max_slices must be >= 0")

    def _select_indices(self, depth: int, stride: int, max_slices: int) -> list[int]:
        indices = list(range(0, depth, stride))
        if max_slices and len(indices) > max_slices:
            indices = indices[:max_slices]
        if not indices:
            raise ValueError(f"No slices selected for depth={depth}")
        return indices

    def _preprocess_batch(self, batch_zyx: np.ndarray, resize: int):
        x = self.torch.from_numpy(batch_zyx)[:, None, ...]
        x = self.torch.clamp(x, min=-1000, max=1000)
        x = (x - self.mean) / (self.std + 1e-8)
        x = x.repeat(1, 3, 1, 1)
        x = self.torch.nn.functional.interpolate(x, size=(resize, resize), mode="bilinear", align_corners=False)
        return x

    def extract_slice_embeddings_from_volume(
        self,
        volume_zyx: np.ndarray,
        cfg: SliceExtractConfig,
        with_stats: bool = True,
    ) -> tuple[np.ndarray, list[int], dict | None]:
        self._validate_extract_config(cfg)
        if volume_zyx.ndim != 3:
            raise ValueError(f"Expected volume [D, H, W], got {volume_zyx.shape}")
        indices = self._select_indices(volume_zyx.shape[0], cfg.stride, cfg.max_slices)

        def _run():
            chunks = []
            with self.torch.inference_mode():
                for i in range(0, len(indices), cfg.batch_size):
                    batch_idx = indices[i : i + cfg.batch_size]
                    batch = np.stack([volume_zyx[j] for j in batch_idx], axis=0)
                    feats = self.model.forward_features(self._preprocess_batch(batch, cfg.resize).to(self.device))
                    chunks.append(feats["x_norm_clstoken"].detach().cpu())
            return self.torch.cat(chunks, dim=0)

        stats = measure_inference(self.torch, self.device, _run) if with_stats else {"result": _run()}
        emb = stats["result"].numpy().astype(np.float32)
        if with_stats:
            clean_stats = {
                "elapsed_s": stats.get("elapsed_s"),
                "gpu_mem_mb": stats.get("gpu_mem_mb"),
                "cpu_rss_mb": stats.get("cpu_rss_mb"),
            }
        else:
            clean_stats = None
        return emb, indices, clean_stats

    def extract_slice_embeddings(
        self,
        input_path: str | Path,
        cfg: SliceExtractConfig,
        with_stats: bool = True,
    ) -> tuple[np.ndarray, list[int], dict | None]:
        volume = load_volume_zyx(input_path)
        return self.extract_slice_embeddings_from_volume(volume, cfg, with_stats=with_stats)

    def similarity_map_from_volume(
        self,
        volume_zyx: np.ndarray,
        slice_index: int,
        resize: int = 224,
        sim_patch: str = "center",
    ) -> np.ndarray | None:
        if not (0 <= slice_index < volume_zyx.shape[0]):
            return None
        slice_img = volume_zyx[slice_index]
        x = self._preprocess_batch(slice_img[None, ...], resize).to(self.device)
        with self.torch.inference_mode():
            feats = self.model.forward_features(x)
            patch_tokens = feats["x_norm_patchtokens"].detach().cpu()[0]
        h = resize // self.patch_size
        w = resize // self.patch_size
        if patch_tokens.shape[0] != h * w:
            return None
        patch_tokens_2d = patch_tokens.reshape(h, w, -1)
        if sim_patch == "center":
            ph, pw = h // 2, w // 2
        else:
            try:
                ph, pw = (int(coord) for coord in sim_patch.split(","))
            except Exception:
                ph, pw = h // 2, w // 2
        ph = max(0, min(h - 1, ph))
        pw = max(0, min(w - 1, pw))
        ref = patch_tokens_2d[ph, pw]
        sim = self.torch.nn.functional.cosine_similarity(
            patch_tokens_2d.reshape(-1, patch_tokens_2d.shape[-1]),
            ref[None, :].repeat(h * w, 1),
            dim=1,
        ).reshape(h, w)
        return sim.numpy().astype(np.float32)

