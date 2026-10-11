"""
Compute descriptive statistics and integrity checks for the dataset.

Writes reports/metrics/dataset_stats.json, which the dashboard and the
report read. Nothing is modified: duplicates and train/test overlap are
counted and reported, never removed.

Usage:  python scripts\\dataset_stats.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinesentinel.config import METRICS_DIR, RAW_DIR, TEST_CSV, TRAIN_CSV  # noqa: E402

OFFICIAL_SOURCE = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
MIRROR = "https://huggingface.co/datasets/stanfordnlp/imdb"


def split_stats(df: pd.DataFrame) -> dict:
    words = df["review"].str.split().str.len()
    return {
        "rows": int(len(df)),
        "labels": {k: int(v) for k, v in df["sentiment"].value_counts().items()},
        "duplicate_rows": int(df["review"].duplicated().sum()),
        "words_mean": float(words.mean()),
        "words_median": float(words.median()),
        "words_min": int(words.min()),
        "words_max": int(words.max()),
        "words_by_label_mean": {k: float(v) for k, v in words.groupby(df["sentiment"]).mean().items()},
    }


def compute_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_texts = set(train_df["review"])
    in_train = test_df["review"].isin(train_texts)
    merged = test_df[in_train].merge(
        train_df.drop_duplicates("review"), on="review", suffixes=("_test", "_train")
    )
    return {
        "train": split_stats(train_df),
        "test": split_stats(test_df),
        "total_rows": int(len(train_df) + len(test_df)),
        "cross_split_overlap": {
            "test_rows_also_in_train": int(in_train.sum()),
            "share_of_test": float(in_train.mean()),
            "label_conflicts": int((merged["sentiment_test"] != merged["sentiment_train"]).sum()),
        },
    }


def main() -> None:
    stats = compute_stats(pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV))
    # scripts/verify_provenance.py records whether the mirror's content matches
    # the official tarball; without that record, no such claim is made.
    provenance_path = METRICS_DIR / "provenance_check.json"
    verified = False
    if provenance_path.exists():
        with open(provenance_path, "r", encoding="utf-8") as f:
            verified = bool(json.load(f).get("all_identical"))
    stats["source"] = {
        "official": OFFICIAL_SOURCE,
        "downloaded_from": MIRROR,
        "verified_against_official": verified,
        "note": (
            "Downloaded from the HuggingFace mirror. Every (review, label) pair in both "
            "splits was compared with the official tarball and found identical."
            if verified
            else "Downloaded from the HuggingFace mirror. Not compared with the official tarball."
        ),
    }
    stats["raw_files"] = {}
    for name in ("train", "test"):
        path = RAW_DIR / f"{name}.parquet"
        if path.exists():
            stats["raw_files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "dataset_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "raw_files"}, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
