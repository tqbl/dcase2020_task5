#!/bin/bash

set -e

evaluate() {
  echo "Evaluating '$1'..."
  python task5/extern/evaluate_predictions.py \
    "_workspace/predictions/$1/seed=1000/validation.csv" \
    '_dataset/annotations.csv' \
    '_dataset/dcase-ust-taxonomy.yaml'
  echo
}

evaluate 'qkcnn10_w1024_max'
evaluate 'qkcnn10_w1024_pseudo'
evaluate 'gcnn_w2048_pseudo'
