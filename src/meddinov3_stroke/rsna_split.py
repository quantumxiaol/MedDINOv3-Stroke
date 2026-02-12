from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .infer_utils import load_env
from .rsna_pipeline import load_series_records, save_split_csv, split_records_by_study, summarize_split


@dataclass
class RSNASplitConfig:
    series_csv: str
    output_dir: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split RSNA series CSV into train/val/test at study level.")
    parser.add_argument("--series-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/rsna_splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: list[str] | None = None) -> RSNASplitConfig:
    args = build_parser().parse_args(argv)
    return RSNASplitConfig(
        series_csv=args.series_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


def run_split(cfg: RSNASplitConfig) -> dict:
    ratio_sum = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"train/val/test ratio sum must be 1.0, got {ratio_sum}")
    records = load_series_records(cfg.series_csv)
    splits = split_records_by_study(
        records=records,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_csv(splits["train"], output_dir / "train.csv", "train")
    save_split_csv(splits["val"], output_dir / "val.csv", "val")
    save_split_csv(splits["test"], output_dir / "test.csv", "test")

    summary = {
        "config": asdict(cfg),
        "train": summarize_split(splits["train"]),
        "val": summarize_split(splits["val"]),
        "test": summarize_split(splits["test"]),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    load_env()
    cfg = parse_args(argv)
    run_split(cfg)
    return 0

