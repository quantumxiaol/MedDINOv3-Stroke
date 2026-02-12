from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import INFER_DEFAULTS, EVAL_DEFAULTS
from .feature_extractor import (
    LABEL_NAMES,
    MedDINOv3FeatureExtractor,
    SliceExtractConfig,
    aggregate_embeddings,
    infer_pool_mode,
    load_volume_zyx,
)
from .head_model import build_head
from .infer_utils import load_env, print_stats


@dataclass
class ICHInferConfig:
    input_path: str
    checkpoint: str
    output_dir: str
    stride: int
    max_slices: int
    batch: int
    resize: int
    top_k: int
    threshold: float
    sim_slice: int
    sim_patch: str
    device: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ICH inference: nii.gz -> hemorrhage probabilities + heatmap.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .nii/.nii.gz")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained head checkpoint (best.pt)")
    parser.add_argument("--output-dir", type=str, default="outputs/ich_infer")

    parser.add_argument("--stride", type=int, default=INFER_DEFAULTS.stride)
    parser.add_argument("--max-slices", type=int, default=INFER_DEFAULTS.max_slices)
    parser.add_argument("--batch", type=int, default=INFER_DEFAULTS.batch_size)
    parser.add_argument("--resize", type=int, default=INFER_DEFAULTS.resize)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sim-slice", type=int, default=-1, help="If <0, auto-choose top suspicious slice.")
    parser.add_argument("--sim-patch", type=str, default=INFER_DEFAULTS.sim_patch)
    parser.add_argument("--device", type=str, default=os.getenv("CT_MODEL_DEVICE", EVAL_DEFAULTS.device))
    return parser


def parse_args(argv: list[str] | None = None) -> ICHInferConfig:
    args = build_parser().parse_args(argv)
    return ICHInferConfig(
        input_path=args.input,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        stride=args.stride,
        max_slices=args.max_slices,
        batch=args.batch,
        resize=args.resize,
        top_k=args.top_k,
        threshold=args.threshold,
        sim_slice=args.sim_slice,
        sim_patch=args.sim_patch,
        device=args.device,
    )


def _validate(cfg: ICHInferConfig) -> None:
    if cfg.batch <= 0:
        raise SystemExit("--batch must be > 0")
    if cfg.resize <= 0:
        raise SystemExit("--resize must be > 0")
    if cfg.stride <= 0:
        raise SystemExit("--stride must be > 0")
    if cfg.max_slices < 0:
        raise SystemExit("--max-slices must be >= 0")
    if cfg.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if not (0.0 < cfg.threshold < 1.0):
        raise SystemExit("--threshold must be in (0,1)")


def _load_head(ckpt_path: Path, torch_module, device):
    ckpt = torch_module.load(ckpt_path, map_location="cpu", weights_only=False)
    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    dropout = float(ckpt.get("dropout", 0.2))
    model = build_head(
        torch_module=torch_module,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, input_dim, num_classes


def run_ich_inference(cfg: ICHInferConfig) -> dict:
    load_env()
    _validate(cfg)
    input_path = Path(cfg.input_path).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    ckpt_path = Path(cfg.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = MedDINOv3FeatureExtractor(device_override=cfg.device)
    extract_cfg = SliceExtractConfig(
        stride=cfg.stride,
        max_slices=cfg.max_slices,
        batch_size=cfg.batch,
        resize=cfg.resize,
    )
    volume = load_volume_zyx(input_path)
    slice_emb, slice_indices, stats = extractor.extract_slice_embeddings_from_volume(volume, extract_cfg, with_stats=True)
    print_stats("ich_infer_backbone", stats or {})

    torch = extractor.torch
    model, ckpt, input_dim, num_classes = _load_head(ckpt_path, torch_module=torch, device=extractor.device)
    if num_classes != len(LABEL_NAMES):
        raise SystemExit(f"Head class count mismatch: expected {len(LABEL_NAMES)}, got {num_classes}")
    pool_mode = infer_pool_mode(input_dim=input_dim, slice_dim=slice_emb.shape[1])

    vol_emb = aggregate_embeddings(slice_emb, mode=pool_mode).astype(np.float32)
    with torch.inference_mode():
        vol_logits = model(torch.from_numpy(vol_emb[None, :]).to(extractor.device))
        vol_probs = torch.sigmoid(vol_logits)[0].detach().cpu().numpy()

    probs = {name: float(vol_probs[i]) for i, name in enumerate(LABEL_NAMES)}
    pred = {name: bool(vol_probs[i] >= cfg.threshold) for i, name in enumerate(LABEL_NAMES)}

    top_slices: list[dict] = []
    if input_dim == slice_emb.shape[1]:
        with torch.inference_mode():
            sl_logits = model(torch.from_numpy(slice_emb).to(extractor.device))
            sl_probs = torch.sigmoid(sl_logits).detach().cpu().numpy()
        any_probs = sl_probs[:, 0]
        top_k = min(cfg.top_k, len(any_probs))
        top_idx = np.argsort(any_probs)[-top_k:][::-1]
        for idx in top_idx:
            top_slices.append(
                {
                    "rank": int(len(top_slices) + 1),
                    "slice_pos": int(idx),
                    "slice_index": int(slice_indices[idx]),
                    "p_any": float(any_probs[idx]),
                }
            )
        np.save(output_dir / "slice_probs.npy", sl_probs.astype(np.float32))

    if cfg.sim_slice >= 0:
        sim_slice = cfg.sim_slice
    elif top_slices:
        sim_slice = int(top_slices[0]["slice_index"])
    else:
        sim_slice = int(slice_indices[len(slice_indices) // 2])
    sim_np = extractor.similarity_map_from_volume(volume, sim_slice, resize=cfg.resize, sim_patch=cfg.sim_patch)

    similarity_map = None
    similarity_png = None
    if sim_np is not None:
        similarity_map = str(output_dir / "similarity_map.npy")
        np.save(similarity_map, sim_np.astype(np.float32))
        try:
            import matplotlib.pyplot as plt

            similarity_png = str(output_dir / "similarity_map.png")
            plt.imsave(similarity_png, sim_np, cmap="viridis")
        except Exception:
            similarity_png = None

    summary = {
        "config": asdict(cfg),
        "input": str(input_path),
        "checkpoint": str(ckpt_path),
        "pool_mode": pool_mode,
        "num_slices": int(len(slice_indices)),
        "slice_embedding_dim": int(slice_emb.shape[1]),
        "probabilities": probs,
        "predicted": pred,
        "threshold": float(cfg.threshold),
        "top_slices": top_slices,
        "similarity_slice_index": int(sim_slice),
        "similarity_map": similarity_map,
        "similarity_map_png": similarity_png,
        "elapsed_s": None if stats is None else stats.get("elapsed_s"),
        "gpu_mem_mb": None if stats is None else stats.get("gpu_mem_mb"),
        "cpu_rss_mb": None if stats is None else stats.get("cpu_rss_mb"),
    }

    np.save(output_dir / "slice_indices.npy", np.asarray(slice_indices, dtype=np.int32))
    np.save(output_dir / "slice_embeddings.npy", slice_emb.astype(np.float32))
    np.save(output_dir / "volume_embedding.npy", vol_emb.astype(np.float32))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    load_env()
    cfg = parse_args(argv)
    run_ich_inference(cfg)
    return 0

