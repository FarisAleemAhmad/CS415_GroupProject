"""Loaders for the processed splits (used later by the training code)."""
from pathlib import Path

import pandas as pd


def load_split(split: str, data_dir: str = "data/processed", limit: int | None = None,
               seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame with columns text, label (0-4), rating (1-5).

    `limit` takes a random subsample (for pilot runs on small data).
    """
    df = pd.read_csv(Path(data_dir) / f"{split}.csv")
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
    return df
