"""Clean Amazon Fine Food Reviews and create product-grouped train/val/test splits.

Usage:
    python -m src.data.prepare --raw data/raw/Reviews.csv --out data/processed

Steps (each is counted and written to data/processed/prep_stats.json):
  1. drop rows with missing/empty review text or invalid score
  2. clean text (strip HTML tags, collapse whitespace)
  3. drop duplicate reviews (identical text), which also prevents the same
     text from landing in different splits
  4. split BY ProductId so all reviews of a product stay in one split
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = HTML_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()


def clean(df: pd.DataFrame):
    stats = {"raw_rows": int(len(df))}

    df = df.dropna(subset=["Text", "Score", "ProductId"]).copy()
    df["Text"] = df["Text"].astype(str).map(clean_text)
    df = df[df["Text"].str.len() > 0]
    df = df[df["Score"].isin([1, 2, 3, 4, 5])]
    stats["after_missing_removed"] = int(len(df))

    df = df.drop_duplicates(subset=["Text"], keep="first")
    stats["after_duplicates_removed"] = int(len(df))

    out = pd.DataFrame(
        {
            "product_id": df["ProductId"].values,
            "text": df["Text"].values,
            "rating": df["Score"].astype(int).values,
        }
    )
    out["label"] = out["rating"] - 1  # 0..4 for models/losses
    out["n_words"] = out["text"].str.split().str.len()
    return out.reset_index(drop=True), stats


def split_by_product(df, seed=42, frac=(0.8, 0.1, 0.1)):
    """Assign whole products to train/val/test, aiming at ~frac of REVIEWS.

    Products are shuffled with a fixed seed, then cumulative review counts
    decide the boundaries, so splits are reproducible and leak-free.
    """
    assert abs(sum(frac) - 1) < 1e-9
    rng = np.random.default_rng(seed)
    counts = df.groupby("product_id").size()
    products = counts.index.to_numpy().copy()  # copy: shuffling a view corrupts the index
    rng.shuffle(products)
    cum = counts.loc[products].cumsum().to_numpy() / len(df)

    split_of = np.where(cum <= frac[0], "train",
                        np.where(cum <= frac[0] + frac[1], "val", "test"))
    mapping = dict(zip(products, split_of))
    df = df.copy()
    df["split"] = df["product_id"].map(mapping)
    return df


def summarize(df):
    out = {}
    for name, g in df.groupby("split"):
        out[name] = {
            "reviews": int(len(g)),
            "products": int(g["product_id"].nunique()),
            "rating_dist": {int(k): round(v, 4) for k, v in
                            g["rating"].value_counts(normalize=True).sort_index().items()},
            "mean_words": round(float(g["n_words"].mean()), 1),
            "median_words": float(g["n_words"].median()),
            "p95_words": float(g["n_words"].quantile(0.95)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/Reviews.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw, usecols=["ProductId", "Score", "Text"])
    df, stats = clean(raw)
    df = split_by_product(df, seed=args.seed)

    # leakage check
    sets = {s: set(g["product_id"]) for s, g in df.groupby("split")}
    for a in sets:
        for b in sets:
            if a < b:
                assert not (sets[a] & sets[b]), f"product overlap {a}/{b}"

    for s, g in df.groupby("split"):
        g.drop(columns="split").to_csv(out_dir / f"{s}.csv", index=False)

    stats["seed"] = args.seed
    stats["splits"] = summarize(df)
    (out_dir / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
