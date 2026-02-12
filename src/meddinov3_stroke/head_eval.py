from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import EVAL_DEFAULTS
from .head_model import build_head, ensure_2d_embeddings, ensure_2d_labels, load_npy, multilabel_auroc, validate_pair
from .infer_utils import load_env, select_device


@dataclass
class EvalConfig:
    embeddings: str
    labels: str
    checkpoint: str
    batch_size: int
    device: str
    output_json: str
    output_probs: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained multi-label head on MedDINOv3 embeddings.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .npy [N, D].")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels .npy [N, C].")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file, usually best.pt.")
    parser.add_argument("--batch-size", type=int, default=EVAL_DEFAULTS.batch_size)
    parser.add_argument("--device", type=str, default=os.getenv("CT_MODEL_DEVICE", EVAL_DEFAULTS.device))
    parser.add_argument("--output-json", type=str, default="", help="Optional metrics output JSON path.")
    parser.add_argument("--output-probs", type=str, default="", help="Optional probabilities .npy output path.")
    return parser


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    args = build_parser().parse_args(argv)
    return EvalConfig(
        embeddings=args.embeddings,
        labels=args.labels,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        output_json=args.output_json,
        output_probs=args.output_probs,
    )


def evaluate_head(cfg: EvalConfig) -> dict:
    load_env()
    if cfg.device:
        os.environ["CT_MODEL_DEVICE"] = cfg.device
    if cfg.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    import torch

    device, device_name = select_device(torch)
    x = ensure_2d_embeddings(load_npy(cfg.embeddings))
    y = ensure_2d_labels(load_npy(cfg.labels))
    validate_pair(x, y)

    ckpt = torch.load(Path(cfg.checkpoint), map_location="cpu", weights_only=False)
    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    dropout = float(ckpt.get("dropout", 0.2))
    if x.shape[1] != input_dim:
        raise SystemExit(f"Embedding dim mismatch: expected {input_dim}, got {x.shape[1]}")
    if y.shape[1] != num_classes:
        raise SystemExit(f"Class count mismatch: expected {num_classes}, got {y.shape[1]}")

    model = build_head(
        torch_module=torch,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pos_weight = ckpt.get("pos_weight")
    if pos_weight is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_loss = 0.0
    n_samples = 0
    all_probs = []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size = xb.shape[0]
            total_loss += float(loss.item()) * batch_size
            n_samples += batch_size
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    auc = multilabel_auroc(y, probs)

    metrics = {
        "config": asdict(cfg),
        "device": device_name,
        "num_samples": int(x.shape[0]),
        "num_classes": int(y.shape[1]),
        "loss": total_loss / max(1, n_samples),
        "macro_auroc": auc.macro,
        "per_class_auroc": auc.per_class,
    }

    print(
        f"[eval] loss={metrics['loss']:.5f} macro_auroc={metrics['macro_auroc']:.5f} "
        f"samples={metrics['num_samples']}"
    )
    if cfg.output_json:
        output_json = Path(cfg.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2))
    if cfg.output_probs:
        output_probs = Path(cfg.output_probs)
        output_probs.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_probs, probs)
    return metrics


def main(argv: list[str] | None = None) -> int:
    load_env()
    cfg = parse_args(argv)
    evaluate_head(cfg)
    return 0
