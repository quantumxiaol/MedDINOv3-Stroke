from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import get_runtime_paths
from .feature_extractor import MedDINOv3FeatureExtractor, SliceExtractConfig, aggregate_embeddings

LABEL_COLUMNS = (
    "label_any",
    "label_epidural",
    "label_intraparenchymal",
    "label_intraventricular",
    "label_subarachnoid",
    "label_subdural",
)


@dataclass(frozen=True)
class RSNASeriesRecord:
    rel_dir: str
    study_uid: str
    series_uid: str
    nifti_path: str
    labels: tuple[int, int, int, int, int, int]
    raw: dict[str, str]


def _to_int01(value: str) -> int:
    try:
        v = int(float(value))
    except Exception as exc:
        raise ValueError(f"Invalid label value: {value}") from exc
    return 1 if v > 0 else 0


def resolve_nifti_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    runtime_paths = get_runtime_paths()
    return (runtime_paths.datasets_dir / path_str).resolve()


def load_series_records(csv_path: str | Path) -> list[RSNASeriesRecord]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    records: list[RSNASeriesRecord] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"rel_dir", "study_uid", "series_uid", "nifti_path", *LABEL_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            labels = tuple(_to_int01(row[col]) for col in LABEL_COLUMNS)
            records.append(
                RSNASeriesRecord(
                    rel_dir=row["rel_dir"],
                    study_uid=row["study_uid"],
                    series_uid=row["series_uid"],
                    nifti_path=row["nifti_path"],
                    labels=labels,  # type: ignore[arg-type]
                    raw=dict(row),
                )
            )
    if not records:
        raise ValueError("No records loaded from CSV.")
    return records


def _compute_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val
    if total >= 3:
        if n_train == 0:
            n_train = 1
            n_test -= 1
        if n_val == 0:
            n_val = 1
            n_test -= 1
        if n_test == 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
    return n_train, n_val, n_test


def split_records_by_study(
    records: list[RSNASeriesRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[RSNASeriesRecord]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")

    study_to_records: dict[str, list[RSNASeriesRecord]] = {}
    for rec in records:
        study_to_records.setdefault(rec.study_uid, []).append(rec)

    study_to_label: dict[str, tuple[int, ...]] = {}
    for study_uid, recs in study_to_records.items():
        label_mat = np.asarray([r.labels for r in recs], dtype=np.int32)
        study_label = tuple(label_mat.max(axis=0).tolist())
        study_to_label[study_uid] = study_label

    strata: dict[tuple[int, ...], list[str]] = {}
    for study_uid, label in study_to_label.items():
        strata.setdefault(label, []).append(study_uid)

    rng = random.Random(seed)
    study_split: dict[str, str] = {}
    for stratum_key, study_uids in strata.items():
        study_uids = list(study_uids)
        rng.shuffle(study_uids)
        n_train, n_val, _ = _compute_counts(len(study_uids), train_ratio, val_ratio)
        for uid in study_uids[:n_train]:
            study_split[uid] = "train"
        for uid in study_uids[n_train : n_train + n_val]:
            study_split[uid] = "val"
        for uid in study_uids[n_train + n_val :]:
            study_split[uid] = "test"

    split_records = {"train": [], "val": [], "test": []}
    for rec in records:
        split_name = study_split[rec.study_uid]
        split_records[split_name].append(rec)

    # Fallback for tiny datasets where per-stratum split can create empty splits.
    if any(len(split_records[key]) == 0 for key in ("train", "val", "test")):
        all_studies = list(study_to_records.keys())
        rng.shuffle(all_studies)
        n_train, n_val, _ = _compute_counts(len(all_studies), train_ratio, val_ratio)
        study_split = {}
        for uid in all_studies[:n_train]:
            study_split[uid] = "train"
        for uid in all_studies[n_train : n_train + n_val]:
            study_split[uid] = "val"
        for uid in all_studies[n_train + n_val :]:
            study_split[uid] = "test"
        split_records = {"train": [], "val": [], "test": []}
        for rec in records:
            split_records[study_split[rec.study_uid]].append(rec)

    if len(study_to_records) >= 3:
        for key in ("train", "val", "test"):
            if not split_records[key]:
                raise ValueError(f"Split {key} is empty; adjust ratios/seed.")
    return split_records


def save_split_csv(records: list[RSNASeriesRecord], out_csv: str | Path, split_name: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError(f"No records to write for split={split_name}")
    fieldnames = list(records[0].raw.keys()) + ["split"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = dict(rec.raw)
            row["split"] = split_name
            writer.writerow(row)


def summarize_split(records: list[RSNASeriesRecord]) -> dict:
    labels = np.asarray([r.labels for r in records], dtype=np.int32)
    return {
        "num_series": int(len(records)),
        "num_studies": int(len({r.study_uid for r in records})),
        "positive_counts": labels.sum(axis=0).tolist(),
    }


def _cache_key(rec: RSNASeriesRecord) -> str:
    return f"{rec.study_uid}__{rec.series_uid}.npy"


def build_embeddings_for_split(
    records: list[RSNASeriesRecord],
    extractor: MedDINOv3FeatureExtractor,
    cache_dir: str | Path,
    extract_cfg: SliceExtractConfig,
    pool_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    emb_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    metas: list[dict] = []

    total = len(records)
    for idx, rec in enumerate(records, 1):
        cache_path = cache_root / _cache_key(rec)
        if cache_path.exists():
            vol_emb = np.load(cache_path).astype(np.float32)
            num_slices = None
        else:
            nii_path = resolve_nifti_path(rec.nifti_path)
            if not nii_path.exists():
                raise FileNotFoundError(f"NIfTI path not found: {nii_path}")
            slice_emb, _, _ = extractor.extract_slice_embeddings(nii_path, extract_cfg, with_stats=False)
            vol_emb = aggregate_embeddings(slice_emb, mode=pool_mode).astype(np.float32)
            np.save(cache_path, vol_emb)
            num_slices = int(slice_emb.shape[0])
        emb_list.append(vol_emb)
        label_list.append(np.asarray(rec.labels, dtype=np.float32))
        metas.append(
            {
                "index": idx,
                "total": total,
                "study_uid": rec.study_uid,
                "series_uid": rec.series_uid,
                "nifti_path": rec.nifti_path,
                "cache": str(cache_path),
                "num_slices": num_slices,
            }
        )
        if idx % 20 == 0 or idx == total:
            print(f"[cache] {idx}/{total}")

    x = np.stack(emb_list, axis=0)
    y = np.stack(label_list, axis=0)
    return x, y, metas
