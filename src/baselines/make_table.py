"""Write results/tables/baselines.tex from results/baselines/baselines.json."""
import json

import numpy as np

d = json.load(open("results/baselines/baselines.json"))
BS = chr(92)


def ms(runs, key):
    v = np.array([r["test"][key] for r in runs])
    return v.mean(), v.std(ddof=1)


def row(label, runs):
    a, af = ms(runs, "accuracy"), ms(runs, "macro_f1")
    m, le = ms(runs, "mae"), ms(runs, "large_error_rate")
    return (f"{label} & {a[0]:.3f} & {af[0]:.3f} & {m[0]:.3f} & "
            f"{le[0]:.3f} " + BS + BS)


mj = d["majority"]["test"]
lines = [
    BS + "begin{tabular}{lcccc}",
    BS + "toprule",
    "Baseline & Accuracy & Macro F1 & MAE & Large Err. " + BS + BS,
    BS + "midrule",
    f"Majority (always 5 stars) & {mj['accuracy']:.3f} & {mj['macro_f1']:.3f} & "
    f"{mj['mae']:.3f} & {mj['large_error_rate']:.3f} " + BS + BS,
    row("TF-IDF, unweighted", d["tfidf_unweighted"]),
    row("TF-IDF, class-balanced", d["tfidf_balanced"]),
    BS + "bottomrule",
    BS + "end{tabular}",
]
open("results/tables/baselines.tex", "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
