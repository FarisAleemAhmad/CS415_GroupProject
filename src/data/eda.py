"""EDA figures and LaTeX table for the report.

Usage: python -m src.data.eda --data data/processed --out results
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SPLITS = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    dfs = {s: pd.read_csv(Path(args.data) / f"{s}.csv") for s in SPLITS}

    # 1. rating distribution per split
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = 0.27
    for i, s in enumerate(SPLITS):
        d = dfs[s]["rating"].value_counts(normalize=True).sort_index()
        ax.bar(d.index + (i - 1) * w, d.values * 100, w, label=s)
    ax.set_xlabel("Star rating")
    ax.set_ylabel("% of reviews")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "rating_distribution.pdf")
    plt.close(fig)

    # 2. review length distribution (train)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(dfs["train"]["n_words"].clip(upper=500), bins=50)
    ax.set_xlabel("Words per review (clipped at 500)")
    ax.set_ylabel("Reviews")
    fig.tight_layout()
    fig.savefig(out / "figures" / "review_length.pdf")
    plt.close(fig)

    # 3. LaTeX split table
    rows = []
    for s in SPLITS:
        d = dfs[s]
        rows.append(f"{s.capitalize()} & {len(d):,} & {d['product_id'].nunique():,} \\\\")
    tex = ("\\begin{tabular}{lrr}\n\\toprule\nSplit & Reviews & Products \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tables" / "splits.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
