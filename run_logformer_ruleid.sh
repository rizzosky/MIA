#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python run_logformer_ruleid.py \
    --dataset ./data/windows_ruleid.pkl --output_dir ./results_ruleid \
    --device mps \
    2>&1 | tee results_ruleid/training_logformer_ruleid.log