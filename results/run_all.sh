#!/bin/bash
# Re-runnable experiment runner. Finished runs (status=complete) are skipped,
# so it is safe to re-launch after a crash or to extend the seed list.
#
#   bash results/run_all.sh                       # TextCNN, 3 losses, seed 0
#   SEEDS="0 1 2" bash results/run_all.sh         # more seeds
#   MODELS="distilbert" bash results/run_all.sh   # DistilBERT (100k subsample, 2 epochs)
cd "$(dirname "$0")/.."
G=data/raw/glove/glove.840B.300d.txt
MODELS=${MODELS:-textcnn}
SEEDS=${SEEDS:-0}
LOSSES=${LOSSES:-"ce oll mll"}
TAG=${TAG:-_v2}
mkdir -p results/logs
for m in $MODELS; do
  for s in $SEEDS; do
    for l in $LOSSES; do
      if [ "$m" = "textcnn" ]; then EXTRA="--epochs 6 --glove $G"; else EXTRA="--epochs 2 --limit 100000"; fi
      python -m src.train --model $m --loss $l --seed $s --tag $TAG $EXTRA >> results/logs/${m}_${l}_s${s}${TAG}.out 2>&1
    done
  done
done
echo ALL_DONE
