"""Shared setup for the side experiments.

Every experiment here works on the TRAINING split only. A stratified 80/20
split of the 25,000 training reviews gives a "fit" part and a "validation"
part; the official test split is never loaded.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from cinesentinel.config import RANDOM_STATE, TRAIN_CSV  # noqa: E402

VALIDATION_FRACTION = 0.2
CLEAN_TRAIN_CSV = TRAIN_CSV.parent / "train_clean.csv"


def load_fit_validation_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split of the cleaned training reviews."""
    if not CLEAN_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{CLEAN_TRAIN_CSV} not found - run `python -m cinesentinel.train` once to build it."
        )
    train_df = pd.read_csv(CLEAN_TRAIN_CSV)
    train_df = train_df[train_df["clean_review"].str.len() > 0].reset_index(drop=True)
    fit_df, val_df = train_test_split(
        train_df,
        test_size=VALIDATION_FRACTION,
        stratify=train_df["sentiment"],
        random_state=RANDOM_STATE,
    )
    return fit_df.reset_index(drop=True), val_df.reset_index(drop=True)
