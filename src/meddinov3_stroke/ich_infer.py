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


def _slice_embeddings_to_head_input(slice_emb: np.ndarray, pool_mode: str) -> np.ndarray:
    if pool_mode == "mean":
        return slice_emb
    if pool_mode == "meanmax":
        return np.concatenate([slice_emb, slice_emb], axis=1)
    raise ValueError(f"Unsupported pool mode: {pool_mode}")


def _single_cls_to_head_input(cls_token, pool_mode: str, torch_module):
    if pool_mode == "mean":
        return cls_token
    if pool_mode == "meanmax":
        return torch_module.cat([cls_token, cls_token], dim=1)
    raise ValueError(f"Unsupported pool mode: {pool_mode}")


def _resize_slice_for_overlay(slice_hw: np.ndarray, resize: int, torch_module) -> np.ndarray:
    raw = torch_module.from_numpy(np.asarray(slice_hw, dtype=np.float32))[None, None, ...]
    raw = torch_module.nn.functional.interpolate(raw, size=(resize, resize), mode="bilinear", align_corners=False)[0, 0]
    raw = torch_module.clamp(raw, min=-1000, max=1000)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    return raw.detach().cpu().numpy()


def _compute_any_saliency_map(
    extractor: MedDINOv3FeatureExtractor,
    head_model,
    slice_hw: np.ndarray,
    resize: int,
    pool_mode: str,
) -> np.ndarray:
    torch = extractor.torch
    x = extractor.preprocess_slice(slice_hw, resize).to(extractor.device)
    x.requires_grad_(True)

    extractor.model.zero_grad(set_to_none=True)
    head_model.zero_grad(set_to_none=True)

    feats = extractor.model.forward_features(x)
    cls_token = feats["x_norm_clstoken"]
    head_in = _single_cls_to_head_input(cls_token, pool_mode, torch_module=torch)
    any_logit = head_model(head_in)[0, 0]
    any_logit.backward()

    if x.grad is None:
        raise RuntimeError("Failed to compute saliency gradient.")
    sal = x.grad.detach()[0].abs().mean(dim=0)
    sal = sal / (sal.max() + 1e-8)
    return sal.detach().cpu().numpy().astype(np.float32)


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

    sl_head_in = _slice_embeddings_to_head_input(slice_emb, pool_mode=pool_mode)
    with torch.inference_mode():
        sl_logits = model(torch.from_numpy(sl_head_in).to(extractor.device))
        sl_probs = torch.sigmoid(sl_logits).detach().cpu().numpy()
    np.save(output_dir / "slice_probs.npy", sl_probs.astype(np.float32))
    any_probs = sl_probs[:, 0]
    top_slices: list[dict] = []
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

    if cfg.sim_slice >= 0:
        lesion_slice = cfg.sim_slice
    elif top_slices:
        lesion_slice = int(top_slices[0]["slice_index"])
    else:
        lesion_slice = int(slice_indices[len(slice_indices) // 2])

    lesion_map = _compute_any_saliency_map(
        extractor=extractor,
        head_model=model,
        slice_hw=volume[lesion_slice],
        resize=cfg.resize,
        pool_mode=pool_mode,
    )
    lesion_heatmap = str(output_dir / "lesion_heatmap.npy")
    np.save(lesion_heatmap, lesion_map.astype(np.float32))

    lesion_heatmap_png = None
    lesion_overlay_png = None
    try:
        import matplotlib.pyplot as plt

        lesion_heatmap_png = str(output_dir / "lesion_heatmap.png")
        plt.imsave(lesion_heatmap_png, lesion_map, cmap="jet")

        base = _resize_slice_for_overlay(volume[lesion_slice], cfg.resize, torch_module=torch)
        lesion_overlay_png = str(output_dir / "lesion_overlay.png")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(base, cmap="gray")
        ax.imshow(lesion_map, cmap="jet", alpha=0.4)
        ax.axis("off")
        fig.savefig(lesion_overlay_png, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except Exception:
        lesion_heatmap_png = None
        lesion_overlay_png = None

    sim_np = extractor.similarity_map_from_volume(volume, lesion_slice, resize=cfg.resize, sim_patch=cfg.sim_patch)

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
        "lesion_slice_index": int(lesion_slice),
        "lesion_heatmap": lesion_heatmap,
        "lesion_heatmap_png": lesion_heatmap_png,
        "lesion_overlay_png": lesion_overlay_png,
        "similarity_slice_index": int(lesion_slice),
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

    # Print concise clinical-facing inference summary to stdout.
    pretty_probs = {k: round(v, 4) for k, v in probs.items()}
    print("[ich] probabilities:", json.dumps(pretty_probs, ensure_ascii=False))
    print("[ich] predicted:", json.dumps(pred, ensure_ascii=False))
    if top_slices:
        print(f"[ich] top suspicious slice: {top_slices[0]}")
    print(f"[ich] lesion heatmap: {output_dir / 'lesion_overlay.png'}")
    print(f"[ich] summary saved: {output_dir / 'summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    load_env()
    cfg = parse_args(argv)
    run_ich_inference(cfg)
    return 0
