"""Non-neural reference baselines on our own splits.

Majority-class predictor and TF-IDF (1-2 grams) + linear classifier, with and
without class weights. Hyperparameters are fixed in advance (no tuning); the
test set is evaluated once per run. Results go to results/baselines/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from src.data.dataset import load_split
from src.evaluation.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = load_split("train", args.data)
    va = load_split("val", args.data)
    te = load_split("test", args.data)
    print("sizes", len(tr), len(va), len(te))
    ytr, yva, yte = (np.asarray(d["label"]) for d in (tr, va, te))
    results = {}

    # majority class
    maj = int(np.bincount(ytr).argmax())
    results["majority"] = {
        "val": compute_metrics(yva, np.full_like(yva, maj)),
        "test": compute_metrics(yte, np.full_like(yte, maj)),
    }

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=60000, min_df=3,
                          sublinear_tf=True, norm="l2")
    Xtr = vec.fit_transform(tr["text"])
    Xva = vec.transform(va["text"])
    Xte = vec.transform(te["text"])
    print("vectorised in %.1fs, %d features" % (time.time() - t0, Xtr.shape[1]))

    for name, cw in [("tfidf_unweighted", None), ("tfidf_balanced", "balanced")]:
        runs = []
        for seed in args.seeds:
            t1 = time.time()
            clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, tol=1e-4,
                                class_weight=cw, random_state=seed, n_jobs=1)
            clf.fit(Xtr, ytr)
            r = {"seed": seed, "fit_s": time.time() - t1,
                 "val": compute_metrics(yva, clf.predict(Xva)),
                 "test": compute_metrics(yte, clf.predict(Xte))}
            runs.append(r)
            print(name, seed, r["val"], r["test"], "%.1fs" % r["fit_s"])
        results[name] = runs

    results["config"] = {"ngram_range": [1, 2], "max_features": 60000, "min_df": 3,
                         "sublinear_tf": True, "alpha": 1e-5, "max_iter": 30,
                         "tol": 1e-4, "seeds": args.seeds,
                         "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    (out / "baselines.json").write_text(json.dumps(results, indent=2))
    print("saved", out / "baselines.json")


if __name__ == "__main__":
    main()
