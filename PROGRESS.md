# Project progress (updated 2026-11-22)

## Status: progress report complete

The progress report (IEEE double-column, 9 pages) is written and compiled in Overleaf. All six settings
(TextCNN, DistilBERT x CE, OLL, MLL) have been run once with seed 0. No `\todo` markers remain.

## What was done
- **Data** (`src/data/`): 568,454 reviews, exact-text duplicates removed (393,135 left), split by product ID
  80/10/10 (seed 42): 314,504 train / 39,171 val / 39,460 test, no product in two splits. About 64% of reviews are 5 stars.
- **Losses** (`src/losses/`): CE, OLL (alpha=1), MLL = lambda*CE + (1-lambda)*OLL (lambda=0.5, from Kasa et al. 2024; not tuned).
- **Models**: TextCNN (frozen GloVe 840B, widths 3/4/5 x 100, 6 epochs, full train set); DistilBERT (full fine-tune,
  lr 3e-5, bs 32, fp16, 2 epochs on a random 100k training subsample).
- **Protocol**: best epoch by validation Macro F1, test set evaluated once. Metrics: accuracy, Macro F1, MAE, large error rate (|error| >= 2).
- **Runs** are stored in `results/runs/<model>_<loss>_s<seed>_v2/` (metrics, predictions, checkpoint). Tables and figures are built by
  `python -m src.evaluation.make_report_assets --tag _v2`.

## Results (seed 0, test set)

| Model | Loss | Acc | Macro F1 | MAE | Large err. |
|---|---|---|---|---|---|
| TextCNN | CE | 0.747 | 0.495 | 0.395 | 0.085 |
| TextCNN | OLL | 0.733 | 0.508 | 0.373 | 0.070 |
| TextCNN | MLL | 0.744 | 0.509 | 0.378 | 0.076 |
| DistilBERT | CE | 0.767 | 0.575 | 0.311 | 0.051 |
| DistilBERT | OLL | 0.746 | 0.556 | 0.320 | 0.048 |
| DistilBERT | MLL | 0.760 | 0.574 | 0.317 | 0.052 |

- TextCNN: both ordinal losses meet the proposal's success criteria (OLL: MAE -5.4%, large errors -17.4%, Macro F1 +0.013).
- DistilBERT: they do not (OLL: MAE +2.8%, large errors -6.7%, Macro F1 -0.018; MLL about equal to CE).
- All numbers are single-seed and preliminary.

## Remaining work (after the progress report)
- 3 seeds per setting, then repeat the success-criteria check on the means.
- Tune lambda, alpha, learning rate and epochs on the validation set with a comparable budget per loss (define the protocol in advance).
- Train DistilBERT on more training data / more epochs to remove the data-budget difference to TextCNN.
- Extended error analysis (mixed-opinion reviews). Optional: cross-category evaluation.
- Optional robustness checks: median decoding, bootstrap confidence intervals, class-weighted CE control.

## Known limitations (also stated in the report)
One seed; DistilBERT subsample and 2 epochs vs. TextCNN full data; loss parameters not tuned; TextCNN may be undertrained
(best epoch was the last); train times are not comparable between runs because the GPU was shared.

## Notes
- Python 3.12, torch 2.5.1+cu121, `transformers<5`, RTX 3070 Ti Laptop 8 GB.
- `results/archive/` holds earlier aggregate-only runs (no predictions saved).
- `overleaf_pkg/`, `overleaf_upload.zip` and `overleaf_upload.pdf` are the Overleaf upload package and the compiled PDF of the submitted report.
