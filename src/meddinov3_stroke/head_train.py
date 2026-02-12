from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import TRAIN_DEFAULTS
from .head_model import (
    build_head,
    compute_pos_weight,
    ensure_2d_embeddings,
    ensure_2d_labels,
    load_npy,
    multilabel_auroc,
    validate_pair,
)
from .infer_utils import load_env, select_device


@dataclass
class TrainConfig:
    train_embeddings: str
    train_labels: str
    val_embeddings: str
    val_labels: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    seed: int
    device: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight multi-label head on MedDINOv3 embeddings.")
    parser.add_argument("--train-embeddings", type=str, required=True, help="Path to train embeddings .npy [N, D].")
    parser.add_argument("--train-labels", type=str, required=True, help="Path to train labels .npy [N, C].")
    parser.add_argument("--val-embeddings", type=str, default="", help="Optional path to validation embeddings .npy.")
    parser.add_argument("--val-labels", type=str, default="", help="Optional path to validation labels .npy.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TRAIN_DEFAULTS.output_dir,
        help="Directory to save checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN_DEFAULTS.lr)
    parser.add_argument("--weight-decay", type=float, default=TRAIN_DEFAULTS.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=TRAIN_DEFAULTS.hidden_dim)
    parser.add_argument("--dropout", type=float, default=TRAIN_DEFAULTS.dropout)
    parser.add_argument("--seed", type=int, default=TRAIN_DEFAULTS.seed)
    parser.add_argument("--device", type=str, default=os.getenv("CT_MODEL_DEVICE", TRAIN_DEFAULTS.device))
    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    return TrainConfig(
        train_embeddings=args.train_embeddings,
        train_labels=args.train_labels,
        val_embeddings=args.val_embeddings,
        val_labels=args.val_labels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _validate_config(cfg: TrainConfig) -> None:
    if cfg.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if cfg.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if cfg.lr <= 0:
        raise SystemExit("--lr must be > 0")
    if cfg.hidden_dim < 0:
        raise SystemExit("--hidden-dim must be >= 0")
    if not (0.0 <= cfg.dropout < 1.0):
        raise SystemExit("--dropout must be in [0, 1)")
    has_val_emb = bool(cfg.val_embeddings)
    has_val_lbl = bool(cfg.val_labels)
    if has_val_emb != has_val_lbl:
        raise SystemExit("--val-embeddings and --val-labels must be provided together.")


def _make_loader(torch_module, embeddings: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    ds = torch_module.utils.data.TensorDataset(
        torch_module.from_numpy(embeddings.astype(np.float32)),
        torch_module.from_numpy(labels.astype(np.float32)),
    )
    return torch_module.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def _run_eval(torch_module, device, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []
    with torch_module.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            n_samples += batch_size
            probs = torch_module.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.detach().cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    auc = multilabel_auroc(y_true, y_prob)
    return {
        "loss": total_loss / max(1, n_samples),
        "macro_auroc": auc.macro,
        "per_class_auroc": auc.per_class,
    }


def train_head(cfg: TrainConfig) -> dict:
    load_env()
    if cfg.device:
        os.environ["CT_MODEL_DEVICE"] = cfg.device
    _validate_config(cfg)
    _set_seed(cfg.seed)

    import torch

    device, device_name = select_device(torch)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = ensure_2d_embeddings(load_npy(cfg.train_embeddings))
    y_train = ensure_2d_labels(load_npy(cfg.train_labels))
    validate_pair(x_train, y_train)

    has_val = bool(cfg.val_embeddings)
    if has_val:
        x_val = ensure_2d_embeddings(load_npy(cfg.val_embeddings))
        y_val = ensure_2d_labels(load_npy(cfg.val_labels))
        validate_pair(x_val, y_val)
        if x_val.shape[1] != x_train.shape[1]:
            raise SystemExit("Validation embedding dimension differs from training embeddings.")
        if y_val.shape[1] != y_train.shape[1]:
            raise SystemExit("Validation class count differs from training labels.")
    else:
        x_val = y_val = None

    model = build_head(
        torch_module=torch,
        input_dim=x_train.shape[1],
        num_classes=y_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    pos_weight = torch.from_numpy(compute_pos_weight(y_train)).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = _make_loader(torch, x_train, y_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _make_loader(torch, x_val, y_val, batch_size=cfg.batch_size, shuffle=False) if has_val else None

    best_metric = None
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            n_samples += batch_size
        train_loss = total_loss / max(1, n_samples)
        row = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            val_metrics = _run_eval(torch, device, model, val_loader, criterion)
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_macro_auroc": val_metrics["macro_auroc"],
                    "val_per_class_auroc": val_metrics["per_class_auroc"],
                }
            )
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.5f} "
                f"val_loss={val_metrics['loss']:.5f} val_macro_auroc={val_metrics['macro_auroc']:.5f}"
            )
            score = val_metrics["macro_auroc"]
            if not np.isfinite(score):
                score = -val_metrics["loss"]
        else:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.5f}")
            score = -train_loss

        history.append(row)
        is_better = best_metric is None or score > best_metric
        if is_better:
            best_metric = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": x_train.shape[1],
                    "num_classes": y_train.shape[1],
                    "hidden_dim": cfg.hidden_dim,
                    "dropout": cfg.dropout,
                    "pos_weight": pos_weight.detach().cpu(),
                    "epoch": epoch,
                    "score": float(score),
                },
                output_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "num_classes": y_train.shape[1],
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "pos_weight": pos_weight.detach().cpu(),
            "epoch": cfg.epochs,
        },
        output_dir / "last.pt",
    )

    summary = {
        "config": asdict(cfg),
        "device": device_name,
        "input_dim": int(x_train.shape[1]),
        "num_classes": int(y_train.shape[1]),
        "num_train": int(x_train.shape[0]),
        "num_val": int(x_val.shape[0]) if x_val is not None else 0,
        "best_score": float(best_metric) if best_metric is not None else None,
        "history": history,
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    load_env()
    cfg = parse_args(argv)
    train_head(cfg)
    return 0
