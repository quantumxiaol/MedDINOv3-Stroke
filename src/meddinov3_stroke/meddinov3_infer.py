from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import INFER_DEFAULTS, get_runtime_paths
from .infer_utils import find_input_nii, load_env, measure_inference, print_stats, select_device


@dataclass
class MedDinoInferConfig:
    input_path: str
    output_dir: str
    stride: int
    max_slices: int
    batch: int
    resize: int
    sim_slice: int
    sim_patch: str
    device: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedDINOv3 feature extraction on CT slices.")
    parser.add_argument("--input", type=str, default="", help="Path to input .nii/.nii.gz (default: first in input/).")
    parser.add_argument("--output-dir", type=str, default=INFER_DEFAULTS.output_dir, help="Output directory.")
    parser.add_argument("--stride", type=int, default=INFER_DEFAULTS.stride)
    parser.add_argument("--max-slices", type=int, default=INFER_DEFAULTS.max_slices)
    parser.add_argument("--batch", type=int, default=INFER_DEFAULTS.batch_size)
    parser.add_argument("--resize", type=int, default=INFER_DEFAULTS.resize)
    parser.add_argument("--sim-slice", type=int, default=INFER_DEFAULTS.sim_slice)
    parser.add_argument("--sim-patch", type=str, default=INFER_DEFAULTS.sim_patch)
    parser.add_argument("--device", type=str, default=os.getenv("CT_MODEL_DEVICE", INFER_DEFAULTS.device))
    return parser


def parse_args(argv: list[str] | None = None) -> MedDinoInferConfig:
    args = build_parser().parse_args(argv)
    return MedDinoInferConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        stride=args.stride,
        max_slices=args.max_slices,
        batch=args.batch,
        resize=args.resize,
        sim_slice=args.sim_slice,
        sim_patch=args.sim_patch,
        device=args.device,
    )


def _validate_args(cfg: MedDinoInferConfig) -> None:
    if cfg.batch <= 0:
        raise SystemExit("--batch must be > 0")
    if cfg.resize <= 0:
        raise SystemExit("--resize must be > 0")
    if cfg.stride <= 0:
        raise SystemExit("--stride must be > 0")
    if cfg.max_slices < 0:
        raise SystemExit("--max-slices must be >= 0")


def _ensure_dinov3_import_path(repo_root: Path) -> None:
    dinov3_parent = (
        repo_root
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


def _load_volume(input_path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(input_path)).get_fdata()
    if img.ndim == 4:
        img = img[..., 0]
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 3:
        raise SystemExit(f"Unexpected input shape: {img.shape}")
    return np.transpose(img, (2, 0, 1))  # D, H, W


def _select_indices(depth: int, stride: int, max_slices: int) -> list[int]:
    indices = list(range(0, depth, stride))
    if max_slices and len(indices) > max_slices:
        indices = indices[:max_slices]
    if not indices:
        raise SystemExit(
            f"No slices selected from volume depth={depth}. "
            "Check --stride/--max-slices configuration."
        )
    return indices


def run_meddinov3_inference(cfg: MedDinoInferConfig) -> dict:
    load_env()
    if cfg.device:
        os.environ["CT_MODEL_DEVICE"] = cfg.device
    _validate_args(cfg)

    paths = get_runtime_paths()
    ckpt_path = paths.meddinov3_ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"MedDINOv3 weights not found: {ckpt_path}")

    repo_root = Path(__file__).resolve().parents[2]
    _ensure_dinov3_import_path(repo_root)

    from dinov3.models.vision_transformer import vit_base
    import torch

    input_path = Path(cfg.input_path).expanduser().resolve() if cfg.input_path else find_input_nii()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = _load_volume(input_path)
    indices = _select_indices(depth=img.shape[0], stride=cfg.stride, max_slices=cfg.max_slices)

    device, _ = select_device(torch)

    model = vit_base(
        drop_path_rate=0.2,
        layerscale_init=1.0e-05,
        n_storage_tokens=4,
        qkv_bias=False,
        mask_k_bias=True,
    )
    chkpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    state = chkpt.get("teacher", chkpt)
    if not isinstance(state, dict):
        raise SystemExit("Unexpected MedDINOv3 checkpoint format.")
    state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if "ibot" not in k and "dino_head" not in k
    }
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    mean = 65.1084213256836
    std = 178.01663208007812

    def preprocess_slices(batch: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(batch)[:, None, ...]
        x = torch.clamp(x, min=-1000, max=1000)
        x = (x - mean) / (std + 1e-8)
        x = x.repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=(cfg.resize, cfg.resize), mode="bilinear", align_corners=False)
        return x

    def _run():
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(indices), cfg.batch):
                batch_idx = indices[i : i + cfg.batch]
                batch = np.stack([img[j] for j in batch_idx], axis=0)
                feats = model.forward_features(preprocess_slices(batch).to(device))
                embeddings.append(feats["x_norm_clstoken"].detach().cpu())
        return torch.cat(embeddings, dim=0)

    stats = measure_inference(torch, device, _run)
    print_stats("meddinov3_vitb16", stats)

    emb = stats["result"]
    np.save(output_dir / "meddinov3_slice_embeddings.npy", emb.numpy())
    np.save(output_dir / "slice_indices.npy", np.asarray(indices, dtype=np.int32))
    np.save(output_dir / "embedding_mean.npy", emb.mean(dim=0).numpy())

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_slices": int(len(indices)),
        "stride": int(cfg.stride),
        "resize": int(cfg.resize),
        "embedding_dim": int(emb.shape[1]),
        "elapsed_s": stats.get("elapsed_s"),
        "gpu_mem_mb": stats.get("gpu_mem_mb"),
        "cpu_rss_mb": stats.get("cpu_rss_mb"),
        "similarity_map": None,
        "similarity_slice_index": None,
    }

    sim_slice = cfg.sim_slice if cfg.sim_slice >= 0 else indices[len(indices) // 2]
    if 0 <= sim_slice < img.shape[0]:
        slice_img = img[sim_slice]
        with torch.inference_mode():
            feats = model.forward_features(preprocess_slices(slice_img[None, ...]).to(device))
            patch_tokens = feats["x_norm_patchtokens"].detach().cpu()[0]
        patch_size = getattr(model, "patch_size", 16)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        h = cfg.resize // patch_size
        w = cfg.resize // patch_size
        if patch_tokens.shape[0] == h * w:
            patch_tokens_2d = patch_tokens.reshape(h, w, -1)
            if cfg.sim_patch == "center":
                ph, pw = h // 2, w // 2
            else:
                try:
                    ph, pw = (int(coord) for coord in cfg.sim_patch.split(","))
                except Exception:
                    ph, pw = h // 2, w // 2
            ph = max(0, min(h - 1, ph))
            pw = max(0, min(w - 1, pw))
            ref = patch_tokens_2d[ph, pw]
            sim = torch.nn.functional.cosine_similarity(
                patch_tokens_2d.reshape(-1, patch_tokens_2d.shape[-1]),
                ref[None, :].repeat(h * w, 1),
                dim=1,
            ).reshape(h, w)
            sim_np = sim.numpy()
            np.save(output_dir / "similarity_map.npy", sim_np)
            summary["similarity_map"] = str(output_dir / "similarity_map.npy")
            summary["similarity_slice_index"] = int(sim_slice)
            try:
                import matplotlib.pyplot as plt

                plt.imsave(output_dir / "similarity_map.png", sim_np, cmap="viridis")
                summary["similarity_map_png"] = str(output_dir / "similarity_map.png")
            except Exception:
                pass

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    load_env()
    cfg = parse_args(argv)
    run_meddinov3_inference(cfg)
    return 0
