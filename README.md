# CS415 Project: Product Review Rating Prediction

Compare cross entropy (CE), Ordinal Log Loss (OLL) and Multitask Log Loss (MLL)
for TextCNN and DistilBERT on Amazon Fine Food Reviews (1-5 star prediction).

Authors: Efe Han Keleş, Ege Aşan, Faris Aleem Ahmad (Sabancı University).

## Layout

```
docs/            proposal PDF, progress report guidelines
data/raw/        original datasets and GloVe (not versioned)
data/processed/  cleaned data and product-ID splits (not versioned)
src/data/        prepare.py (cleaning, split), eda.py, dataset.py
src/models/      textcnn.py, distilbert.py
src/losses/      losses.py (CE, OLL, MLL)
src/evaluation/  metrics.py, make_report_assets.py (tables and figures)
src/train.py     training / evaluation of one run
results/runs/    one folder per run: metrics.json, val/test predictions, model.pt
results/archive/ earlier aggregate-only results
results/logs/    console logs of the runs
results/tables/  LaTeX tables used by the report
results/figures/ figures used by the report
results/run_all.sh  runs all model x loss x seed combinations
report/          IEEE LaTeX progress report (main.tex, sections/, references.bib)
```

## Reproducing the results

```
python -m src.data.prepare                 # clean, deduplicate, split by product
python -m src.data.eda                     # EDA figures, split table
bash results/run_all.sh                    # TextCNN, seed 0, tag _v2
MODELS=distilbert bash results/run_all.sh  # DistilBERT (100k subsample, 2 epochs)
SEEDS="0 1 2" bash results/run_all.sh      # more seeds; completed runs are skipped
python -m src.evaluation.make_report_assets --tag _v2
```

TextCNN needs `data/raw/glove/glove.840B.300d.txt`. See `requirements.txt` for
packages (`transformers` must stay below version 5 with PyTorch 2.5.1).

## Report

Build `report/main.tex` (IEEEtran, conference mode, pdfLaTeX + BibTeX). The report reads
`../results/tables/*.tex` and `../results/figures/*`, so on Overleaf upload both
`report/` and `results/` with the same relative layout and set `report/main.tex`
as the main document.
