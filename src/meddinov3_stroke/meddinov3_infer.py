from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import INFER_DEFAULTS
from .feature_extractor import MedDINOv3FeatureExtractor, SliceExtractConfig, load_volume_zyx
from .infer_utils import find_input_nii, load_env, print_stats


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


def run_meddinov3_inference(cfg: MedDinoInferConfig) -> dict:
    load_env()
    _validate_args(cfg)

    input_path = Path(cfg.input_path).expanduser().resolve() if cfg.input_path else find_input_nii()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
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
    emb, indices, stats = extractor.extract_slice_embeddings_from_volume(volume, extract_cfg, with_stats=True)
    print_stats("meddinov3_vitb16", stats or {})

    np.save(output_dir / "meddinov3_slice_embeddings.npy", emb)
    np.save(output_dir / "slice_indices.npy", np.asarray(indices, dtype=np.int32))
    np.save(output_dir / "embedding_mean.npy", emb.mean(axis=0))

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_slices": int(len(indices)),
        "stride": int(cfg.stride),
        "resize": int(cfg.resize),
        "embedding_dim": int(emb.shape[1]),
        "elapsed_s": None if stats is None else stats.get("elapsed_s"),
        "gpu_mem_mb": None if stats is None else stats.get("gpu_mem_mb"),
        "cpu_rss_mb": None if stats is None else stats.get("cpu_rss_mb"),
        "similarity_map": None,
        "similarity_slice_index": None,
    }

    sim_slice = cfg.sim_slice if cfg.sim_slice >= 0 else indices[len(indices) // 2]
    sim_np = extractor.similarity_map_from_volume(volume, sim_slice, resize=cfg.resize, sim_patch=cfg.sim_patch)
    if sim_np is not None:
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

