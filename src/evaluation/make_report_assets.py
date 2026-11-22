"""Build report tables and figures from results/runs/*.

Usage:
    python -m src.evaluation.make_report_assets --tag _v2

Outputs (missing runs are skipped, so it works on partial results):
    results/tables/main_results.tex          rows = model x loss; mean +- std if several seeds
    results/figures/confusion_<model>.pdf    row-normalised confusion matrices (CE | OLL | MLL)
    results/figures/error_distance_<model>.pdf   share of predictions off by 0..4 stars
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = [("textcnn", "TextCNN"), ("distilbert", "DistilBERT")]
LOSSES = [("ce", "CE"), ("oll", "OLL"), ("mll", "MLL")]
METRICS = ["accuracy", "macro_f1", "mae", "large_error_rate"]


def load_runs(runs_dir, tag):
    """{(model, loss): [run dict with metrics + predictions, ...]} for complete runs."""
    out = {}
    for m, _ in MODELS:
        for l, _ in LOSSES:
            for d in sorted(Path(runs_dir).glob(f"{m}_{l}_s*{tag}")):
                mp = d / "metrics.json"
                if not mp.exists():
                    continue
                meta = json.loads(mp.read_text())
                if meta.get("status") != "complete" or not (d / "test_preds.npz").exists():
                    continue
                p = np.load(d / "test_preds.npz")
                out.setdefault((m, l), []).append({"meta": meta, "y": p["y_true"], "p": p["y_pred"]})
    return out


def fmt(vals, digits=3):
    a = np.asarray(vals, dtype=float)
    pm = r" $\pm$ "
    return f"{a.mean():.{digits}f}" if len(a) == 1 else f"{a.mean():.{digits}f}{pm}{a.std(ddof=1):.{digits}f}"


def main_table(runs, out_path):
    rows = []
    for mi, (m, mname) in enumerate(MODELS):
        present = [(l, lname) for l, lname in LOSSES if (m, l) in runs]
        for j, (l, lname) in enumerate(present):
            rs = runs[(m, l)]
            cells = [fmt([r["meta"]["test"][k] for r in rs]) for k in METRICS]
            minutes = fmt([r["meta"]["train_time_s"] / 60 for r in rs], 1)
            first = (r"\multirow{" + str(len(present)) + r"}{*}{" + mname + "}") if j == 0 else ""
            rows.append(f"{first} & {lname} & {len(rs)} & " + " & ".join(cells) + f" & {minutes} " + r"\\")
        later_has_runs = any((m2, l2) in runs for m2, _ in MODELS[mi + 1:] for l2, _ in LOSSES)
        if present and later_has_runs:
            rows.append(r"\midrule")
    header = "Model & Loss & Seeds & Accuracy & Macro F1 & MAE & Large Err. & Train (min) " + r"\\"
    tex = (r"\begin{tabular}{llcccccc}" + "\n" + r"\toprule" + "\n" + header + "\n" + r"\midrule" + "\n"
           + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    out_path.write_text(tex)


def confusion_fig(runs, m, mname, out_path):
    present = [(l, ln) for l, ln in LOSSES if (m, l) in runs]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(7.0, 2.5), squeeze=False)
    for ax, (l, ln) in zip(axes[0], present):
        r = runs[(m, l)][0]  # first seed
        cm = np.zeros((5, 5))
        for t, p in zip(r["y"], r["p"]):
            cm[t, p] += 1
        cm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6), fontsize=7)
        ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6), fontsize=7)
        ax.set_xlabel("Predicted rating", fontsize=8)
        ax.set_title(f"{mname} / {ln}", fontsize=8)
    axes[0][0].set_ylabel("True rating", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def error_distance_fig(runs, m, mname, out_path):
    present = [(l, ln) for l, ln in LOSSES if (m, l) in runs]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = 0.8 / len(present)
    for k, (l, ln) in enumerate(present):
        shares = np.mean([[np.mean(np.abs(r["y"] - r["p"]) == d) for d in range(5)] for r in runs[(m, l)]], axis=0)
        ax.bar(np.arange(5) + (k - (len(present) - 1) / 2) * w, shares * 100, w, label=ln)
    ax.set_yscale("log")
    ax.set_xlabel("Absolute error (stars)", fontsize=8)
    ax.set_ylabel("% of test reviews (log)", fontsize=8)
    ax.set_title(mname, fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--tag", default="_v2")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs, args.tag)
    print("complete runs:", {f"{m}/{l}": len(v) for (m, l), v in runs.items()})
    main_table(runs, out / "tables" / "main_results.tex")
    for m, mname in MODELS:
        confusion_fig(runs, m, mname, out / "figures" / f"confusion_{m}.pdf")
        error_distance_fig(runs, m, mname, out / "figures" / f"error_distance_{m}.pdf")


if __name__ == "__main__":
    main()
