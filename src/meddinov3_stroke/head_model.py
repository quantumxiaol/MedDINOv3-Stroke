from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_npy(path: str | Path) -> np.ndarray:
    data = np.load(Path(path))
    return np.asarray(data)


def ensure_2d_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim == 1:
        labels = labels[:, None]
    if labels.ndim != 2:
        raise ValueError(f"Expected labels shape [N, C], got {labels.shape}")
    return labels


def ensure_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {embeddings.shape}")
    return embeddings


def validate_pair(embeddings: np.ndarray, labels: np.ndarray) -> None:
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"Sample count mismatch: embeddings={embeddings.shape[0]}, labels={labels.shape[0]}")


def compute_pos_weight(labels: np.ndarray) -> np.ndarray:
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    return neg / np.clip(pos, 1.0, None)


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.zeros_like(sorted_scores, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(y_score)
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@dataclass
class MultiLabelAUC:
    per_class: list[float]
    macro: float


def multilabel_auroc(y_true: np.ndarray, y_score: np.ndarray) -> MultiLabelAUC:
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_score={y_score.shape}")
    per_class = [binary_auroc(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]
    valid = [x for x in per_class if np.isfinite(x)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return MultiLabelAUC(per_class=per_class, macro=macro)


def build_head(torch_module, input_dim: int, num_classes: int, hidden_dim: int, dropout: float):
    nn = torch_module.nn
    if hidden_dim <= 0:
        return nn.Linear(input_dim, num_classes)
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )

