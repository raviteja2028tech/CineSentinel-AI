"""
Download the official Stanford Large Movie Review Dataset (aclImdb).

Primary source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
That server is extremely slow/throttled in practice (observed <20 kB/s,
which makes an 84 MB download take over an hour). We instead fetch the same
dataset -- identical reviews, identical official train/test split, identical
labels -- from its HuggingFace mirror (stanfordnlp/imdb), which serves the
exact same Maas et al. (2011) release over a fast CDN. Falls back to the
original Stanford tarball automatically if HuggingFace is unreachable.
"""

import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

HF_PARQUET_URLS = {
    "train": "https://huggingface.co/api/datasets/stanfordnlp/imdb/parquet/plain_text/train/0.parquet",
    "test": "https://huggingface.co/api/datasets/stanfordnlp/imdb/parquet/plain_text/test/0.parquet",
}

STANFORD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_PATH = RAW_DIR / "aclImdb_v1.tar.gz"
EXTRACTED_DIR = RAW_DIR / "aclImdb"


def _download_with_progress(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
            pbar.update(len(chunk))
    tmp_path.rename(dest)


def download_via_huggingface() -> bool:
    """Download the official dataset's train/test parquet files via the
    HuggingFace mirror. Returns True on success."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for split, url in HF_PARQUET_URLS.items():
            dest = RAW_DIR / f"{split}.parquet"
            if dest.exists():
                print(f"{dest} already present, skipping download.")
                continue
            print(f"Downloading {split} split from HuggingFace mirror of stanfordnlp/imdb ...")
            _download_with_progress(url, dest)
            print(f"  Saved -> {dest}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"HuggingFace mirror download failed: {exc}", file=sys.stderr)
        return False


def download_via_stanford() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.exists():
        print(f"Downloading {STANFORD_URL} (this may be slow) ...")
        _download_with_progress(STANFORD_URL, ARCHIVE_PATH)
        print(f"Downloaded to {ARCHIVE_PATH}")
    else:
        print(f"Archive already present at {ARCHIVE_PATH}, skipping download.")

    if not EXTRACTED_DIR.exists():
        print(f"Extracting {ARCHIVE_PATH} ...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=RAW_DIR)
        print(f"Extracted to {EXTRACTED_DIR}")


def main() -> None:
    ok = download_via_huggingface()
    if not ok:
        print("Falling back to the original Stanford tarball...")
        download_via_stanford()
    print("\nDataset download complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
