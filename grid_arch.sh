#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p results/grid_arch

python grid_search.py --dataset data/windows.pkl --device mps \
    --arch --output_dir results/grid_arch \
        2>&1 | tee results/grid_arch/grid_arch.log