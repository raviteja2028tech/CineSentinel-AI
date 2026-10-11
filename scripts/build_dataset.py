"""
Build train.csv / test.csv from the downloaded parquet files.

These parquet files are the HuggingFace mirror (stanfordnlp/imdb) of the
official Stanford aclImdb dataset: the same 50,000 reviews, the same
official train/test split (25,000 / 25,000, each 50% positive / 50%
negative), and the same binary labels used in the reference paper
(label 1 = rating >= 7 "positive", label 0 = rating <= 4 "negative";
reviews with ratings 5-6 are excluded from the labeled set entirely).
We preserve this split exactly rather than re-shuffling or re-splitting.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTRACTED_DIR = RAW_DIR / "aclImdb"

LABEL_MAP = {0: "negative", 1: "positive"}


def load_split_from_tarball_dir(split: str, extracted_dir: Path) -> pd.DataFrame:
    """Read one split from the extracted official tarball (aclImdb/<split>/<pos|neg>/*.txt)."""
    rows = []
    for folder, sentiment in (("neg", "negative"), ("pos", "positive")):
        for path in sorted((extracted_dir / split / folder).glob("*.txt")):
            rows.append({"review": path.read_text(encoding="utf-8"), "sentiment": sentiment})
    df = pd.DataFrame(rows, columns=["review", "sentiment"])
    df["split"] = split
    return df


def load_split(split: str) -> pd.DataFrame:
    parquet_path = RAW_DIR / f"{split}.parquet"
    if not parquet_path.exists():
        return load_split_from_tarball_dir(split, EXTRACTED_DIR)
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={"text": "review"})
    df["sentiment"] = df["label"].map(LABEL_MAP)
    df["split"] = split
    return df[["review", "sentiment", "split"]]


def main() -> None:
    for split in ("train", "test"):
        parquet_path = RAW_DIR / f"{split}.parquet"
        if not parquet_path.exists() and not (EXTRACTED_DIR / split).exists():
            print(
                f"ERROR: neither {parquet_path} nor {EXTRACTED_DIR / split} found. "
                "Run scripts/download_data.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        print(f"Building {split} split...")
        df = load_split(split)
        assert len(df) == 25000, f"Expected 25000 rows for {split}, got {len(df)}"
        assert (df["sentiment"] == "positive").sum() == 12500
        assert (df["sentiment"] == "negative").sum() == 12500

        # Shuffle deterministically; the source file is label-grouped
        # (all negatives then all positives), which would otherwise bias
        # the ordering seen by downstream cross-validation folds.
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        out_path = PROCESSED_DIR / f"{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path}")
        print(f"  Class balance: {df['sentiment'].value_counts().to_dict()}")

    print("Dataset build complete. Official train/test split preserved (25,000 / 25,000).")


if __name__ == "__main__":
    main()
