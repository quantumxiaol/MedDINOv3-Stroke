from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import INFER_DEFAULTS, TRAIN_DEFAULTS
from .feature_extractor import MedDINOv3FeatureExtractor, SliceExtractConfig
from .head_eval import EvalConfig, evaluate_head
from .head_train import TrainConfig, train_head
from .infer_utils import load_env
from .rsna_pipeline import (
    build_embeddings_for_split,
    load_series_records,
    save_split_csv,
    split_valid_invalid_records,
    split_records_by_study,
    summarize_split,
)


@dataclass
class RSNATrainPipelineConfig:
    series_csv: str
    output_dir: str
    cache_dir: str
    checkpoints_dir: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    pool_mode: str
    stride: int
    max_slices: int
    batch: int
    resize: int
    device: str
    epochs: int
    train_batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    skip_invalid_nifti: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RSNA ICH head from series CSV and NIfTI paths.")
    parser.add_argument("--series-csv", type=str, required=True, help="Path to rsna_hemorrhage_series_labels.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/rsna_pipeline")
    parser.add_argument("--cache-dir", type=str, default="outputs/rsna_cache")
    parser.add_argument("--checkpoints-dir", type=str, default=TRAIN_DEFAULTS.output_dir)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=TRAIN_DEFAULTS.seed)
    parser.add_argument("--pool-mode", type=str, choices=["mean", "meanmax"], default="mean")

    parser.add_argument("--stride", type=int, default=INFER_DEFAULTS.stride)
    parser.add_argument("--max-slices", type=int, default=INFER_DEFAULTS.max_slices)
    parser.add_argument("--batch", type=int, default=INFER_DEFAULTS.batch_size)
    parser.add_argument("--resize", type=int, default=INFER_DEFAULTS.resize)
    parser.add_argument("--device", type=str, default=os.getenv("CT_MODEL_DEVICE", TRAIN_DEFAULTS.device))

    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS.epochs)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN_DEFAULTS.lr)
    parser.add_argument("--weight-decay", type=float, default=TRAIN_DEFAULTS.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=TRAIN_DEFAULTS.hidden_dim)
    parser.add_argument("--dropout", type=float, default=TRAIN_DEFAULTS.dropout)
    parser.add_argument(
        "--skip-invalid-nifti",
        action="store_true",
        help="Skip invalid nifti_path records instead of stopping.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> RSNATrainPipelineConfig:
    args = build_parser().parse_args(argv)
    return RSNATrainPipelineConfig(
        series_csv=args.series_csv,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        checkpoints_dir=args.checkpoints_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        pool_mode=args.pool_mode,
        stride=args.stride,
        max_slices=args.max_slices,
        batch=args.batch,
        resize=args.resize,
        device=args.device,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        skip_invalid_nifti=args.skip_invalid_nifti,
    )


def _validate(cfg: RSNATrainPipelineConfig) -> None:
    if cfg.train_ratio <= 0 or cfg.val_ratio <= 0 or cfg.test_ratio <= 0:
        raise SystemExit("train/val/test ratios must be > 0")
    ratio_sum = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"train/val/test ratio sum must be 1.0, got {ratio_sum}")
    if cfg.batch <= 0 or cfg.train_batch_size <= 0:
        raise SystemExit("batch sizes must be > 0")
    if cfg.epochs <= 0:
        raise SystemExit("epochs must be > 0")
    if cfg.resize <= 0:
        raise SystemExit("resize must be > 0")
    if cfg.hidden_dim < 0:
        raise SystemExit("hidden_dim must be >= 0")


def _save_arrays(root: Path, split_name: str, x: np.ndarray, y: np.ndarray) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    emb_path = root / f"{split_name}_embeddings.npy"
    lbl_path = root / f"{split_name}_labels.npy"
    np.save(emb_path, x.astype(np.float32))
    np.save(lbl_path, y.astype(np.float32))
    return str(emb_path), str(lbl_path)


def run_pipeline(cfg: RSNATrainPipelineConfig) -> dict:
    load_env()
    _validate(cfg)

    output_dir = Path(cfg.output_dir)
    split_dir = output_dir / "splits"
    feature_dir = output_dir / "features"
    cache_dir = Path(cfg.cache_dir)
    shared_cache_dir = cache_dir / "volume_embeddings"
    checkpoints_dir = Path(cfg.checkpoints_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    records = load_series_records(cfg.series_csv)
    splits = split_records_by_study(
        records=records,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )
    save_split_csv(splits["train"], split_dir / "train.csv", "train")
    save_split_csv(splits["val"], split_dir / "val.csv", "val")
    save_split_csv(splits["test"], split_dir / "test.csv", "test")

    invalid_report = {}
    for split_name in ("train", "val", "test"):
        valid, invalid = split_valid_invalid_records(splits[split_name])
        if invalid:
            invalid_report[split_name] = {
                "count": len(invalid),
                "examples": invalid[:10],
            }
            print(f"[warn] split={split_name} invalid_nifti={len(invalid)}")
            if not cfg.skip_invalid_nifti:
                example = invalid[0]
                raise SystemExit(
                    "Invalid NIfTI path found before training. "
                    "Use --skip-invalid-nifti to continue. "
                    f"Example: split={split_name}, study_uid={example['study_uid']}, "
                    f"series_uid={example['series_uid']}, resolved={example['resolved_path']}, "
                    f"reason={example['reason']}"
                )
        splits[split_name] = valid
        if not splits[split_name]:
            raise SystemExit(f"Split {split_name} has no valid records after filtering.")

    print("[pipeline] loading MedDINOv3 backbone once and caching volume embeddings...")
    extractor = MedDINOv3FeatureExtractor(device_override=cfg.device)
    extract_cfg = SliceExtractConfig(
        stride=cfg.stride,
        max_slices=cfg.max_slices,
        batch_size=cfg.batch,
        resize=cfg.resize,
    )

    train_x, train_y, _ = build_embeddings_for_split(
        splits["train"],
        extractor=extractor,
        cache_dir=shared_cache_dir,
        extract_cfg=extract_cfg,
        pool_mode=cfg.pool_mode,
        skip_invalid_nifti=cfg.skip_invalid_nifti,
    )
    val_x, val_y, _ = build_embeddings_for_split(
        splits["val"],
        extractor=extractor,
        cache_dir=shared_cache_dir,
        extract_cfg=extract_cfg,
        pool_mode=cfg.pool_mode,
        skip_invalid_nifti=cfg.skip_invalid_nifti,
    )
    test_x, test_y, _ = build_embeddings_for_split(
        splits["test"],
        extractor=extractor,
        cache_dir=shared_cache_dir,
        extract_cfg=extract_cfg,
        pool_mode=cfg.pool_mode,
        skip_invalid_nifti=cfg.skip_invalid_nifti,
    )

    train_emb, train_lbl = _save_arrays(feature_dir, "train", train_x, train_y)
    val_emb, val_lbl = _save_arrays(feature_dir, "val", val_x, val_y)
    test_emb, test_lbl = _save_arrays(feature_dir, "test", test_x, test_y)

    train_cfg = TrainConfig(
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        output_dir=str(checkpoints_dir),
        epochs=cfg.epochs,
        batch_size=cfg.train_batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        seed=cfg.seed,
        device=cfg.device,
    )
    train_summary = train_head(train_cfg)

    best_ckpt = Path(train_summary["best_checkpoint"])
    eval_cfg = EvalConfig(
        embeddings=test_emb,
        labels=test_lbl,
        checkpoint=str(best_ckpt),
        batch_size=256,
        device=cfg.device,
        output_json=str(output_dir / "test_metrics.json"),
        output_probs=str(output_dir / "test_probs.npy"),
    )
    test_metrics = evaluate_head(eval_cfg)

    pipeline_summary = {
        "config": asdict(cfg),
        "splits": {
            "train": summarize_split(splits["train"]),
            "val": summarize_split(splits["val"]),
            "test": summarize_split(splits["test"]),
        },
        "features": {
            "train_embeddings": train_emb,
            "train_labels": train_lbl,
            "val_embeddings": val_emb,
            "val_labels": val_lbl,
            "test_embeddings": test_emb,
            "test_labels": test_lbl,
        },
        "train_summary": train_summary,
        "test_metrics": test_metrics,
        "invalid_nifti_report": invalid_report,
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, indent=2))
    return pipeline_summary


def main(argv: list[str] | None = None) -> int:
    load_env()
    cfg = parse_args(argv)
    run_pipeline(cfg)
    return 0
