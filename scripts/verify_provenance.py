"""
Compare the project's dataset with the official Stanford tarball.

The project downloads the dataset from a HuggingFace mirror. This script
checks that the mirror's content is the official release: for each split it
compares every (review text, label) pair in data/processed/<split>.csv with
the labelled reviews inside aclImdb_v1.tar.gz.

The tarball is read in place; nothing is extracted and the project's data
files are not modified.

Usage:
    python scripts\\verify_provenance.py --tarball path\\to\\aclImdb_v1.tar.gz

Official tarball: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Output: reports/metrics/provenance_check.json
"""

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinesentinel.config import METRICS_DIR, TEST_CSV, TRAIN_CSV  # noqa: E402

FOLDER_LABELS = {"pos": "positive", "neg": "negative"}


def read_tarball(tarball: Path) -> dict:
    """{split: Counter of (text, label)} for the labelled reviews in the tarball."""
    pairs = {"train": Counter(), "test": Counter()}
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            # aclImdb/<split>/<pos|neg>/<id>_<rating>.txt
            if len(parts) != 4 or parts[1] not in pairs or parts[2] not in FOLDER_LABELS:
                continue
            if not member.isfile() or not parts[3].endswith(".txt"):
                continue
            text = tar.extractfile(member).read().decode("utf-8")
            pairs[parts[1]][(text, FOLDER_LABELS[parts[2]])] += 1
    return pairs


def compare(project: pd.DataFrame, official: Counter) -> dict:
    ours = Counter(zip(project["review"], project["sentiment"]))
    return {
        "project_rows": int(sum(ours.values())),
        "official_rows": int(sum(official.values())),
        "rows_only_in_project": int(sum((ours - official).values())),
        "rows_only_in_official": int(sum((official - ours).values())),
        "identical": ours == official,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tarball", required=True, type=Path, help="Path to aclImdb_v1.tar.gz")
    args = parser.parse_args()

    print(f"Reading {args.tarball} ...")
    official = read_tarball(args.tarball)

    result = {
        "tarball_bytes": args.tarball.stat().st_size,
        "tarball_sha256": hashlib.sha256(args.tarball.read_bytes()).hexdigest(),
        "splits": {
            "train": compare(pd.read_csv(TRAIN_CSV), official["train"]),
            "test": compare(pd.read_csv(TEST_CSV), official["test"]),
        },
    }
    result["all_identical"] = all(s["identical"] for s in result["splits"].values())

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "provenance_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved -> {out_path}")
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
