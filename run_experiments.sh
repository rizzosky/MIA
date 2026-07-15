#!/bin/bash

mkdir -p results
export PYTORCH_ENABLE_MPS_FALLBACK=1

python run_experiments.py \
    --dataset     ./data/windows.pkl \
    --output_dir  ./results \
    --device      mps \
    --models      transformer,bert,deeplog \
    2>&1 | tee results/training_v2.log
