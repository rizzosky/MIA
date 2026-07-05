#!/bin/bash

# Los 3 modelos base
export PYTORCH_ENABLE_MPS_FALLBACK=1 
python run_experiments.py \
    --dataset     ./data/windows_pfsense_500k.pkl \
    --output_dir  ./results_pfsense_500k \
    --device      mps \
    --models      transformer,bert,deeplog \
    2>&1 | tee results_pfsense_500k/training_pfsense.log

# LogFormer
#python run_logformer.py \
#    --dataset    ./data/windows_pfsense_500k.pkl \
#    --output_dir ./results_pfsense_500k \
#    --device     mps \
#    2>&1 | tee results_pfsense_500k/training_logformer_pfsense.log