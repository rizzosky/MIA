#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p results/grid

python grid_search.py --dataset data/windows.pkl --device mps \
        --models transformer,deeplog,bert --output_dir results/grid \
        2>&1 | tee results/grid/grid_search.log