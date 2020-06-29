#!/bin/bash

set -e

extract() {
  echo "Extracting features (n_fft=$1)..."
  python task5/extract.py -f "scripts/w$1_max.conf" 'training'
  python task5/extract.py -f "scripts/w$1_max.conf" 'validation'
  python task5/extract.py -f "scripts/w$1_max.conf" 'test'
}

trial() {
  fname=$(basename "$1")
  id="$2_${fname%.*}/seed=$3"

  echo -e "\nTraining for '$id'..."
  python task5/train.py -f "$1" --model $2 --seed $3 --training_id "$id"

  echo -e "\nPredicting for '$id'..."
  python task5/predict.py -f "$1" 'validation' --training_id "$id" --clean=True
}

experiment() {
    # Currently runs a single trial
    trial "$1" $2 1000
}

extract 1024
extract 2048

experiment 'scripts/w1024_max.conf' 'qkcnn10'
experiment 'scripts/w2048_max.conf' 'qkcnn10'
experiment 'scripts/w1024_pseudo.conf' 'qkcnn10'
experiment 'scripts/w2048_pseudo.conf' 'qkcnn10'
experiment 'scripts/w2048_pseudo.conf' 'gcnn'
