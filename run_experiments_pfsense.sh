#!/bin/bash

mkdir -p results_pfsense_500k_fixed
# Los 3 modelos base
export PYTORCH_ENABLE_MPS_FALLBACK=1 
python run_experiments.py \
    --dataset     ./data/windows_pfsense_500k.pkl \
    --output_dir  ./results_pfsense_500k_fixed \
    --device      mps \
    --max_seq_len 64 \
    --models      transformer,bert,deeplog \
    2>&1 | tee results_pfsense_500k_fixed/training_pfsense.log

# LogFormer
python run_logformer.py \
    --dataset    ./data/windows_pfsense_500k.pkl \
    --output_dir ./results_pfsense_500k_fixed \
    --max_seq_len 64 \
    --device     mps \
    2>&1 | tee results_pfsense_500k_fixed/training_logformer_pfsense.log