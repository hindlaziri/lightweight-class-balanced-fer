#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 src/train.py --data data_train.pt --out results/proposed --epochs 10 --batch-size 256 --image-size 96 --loss cbfocal --label-smoothing 0.05 --gamma 2.0 --seed 42 --num-workers 0 --no-pretrained
python3 src/train.py --data data_train.pt --out results/ce --epochs 10 --batch-size 256 --image-size 96 --loss ce --label-smoothing 0.05 --seed 42 --num-workers 0 --no-pretrained
python3 src/evaluate.py --data data_train.pt --checkpoint results/proposed/best.pt --out results/proposed/evaluation --batch-size 256 --num-workers 0 --seed 42
python3 src/evaluate.py --data data_train.pt --checkpoint results/ce/best.pt --out results/ce/evaluation --batch-size 256 --num-workers 0 --seed 42
python3 src/make_figures.py --data data_train.pt --run results/proposed --out figures
