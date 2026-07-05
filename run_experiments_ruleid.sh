#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1

python run_experiments_ruleid.py \
    --dataset    ./data/windows_ruleid.pkl \
    --output_dir ./results_ruleid \
    --device     mps \
    --models     transformer_logkey,deeplog_logkey \
    2>&1 | tee results_ruleid/training_ruleid.log