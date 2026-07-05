#!/bin/bash

# Correr LogFormer guardando el log
python run_logformer.py \
    --dataset    ./data/windows.pkl \
    --output_dir ./results \
    --device     mps \
    --pretrain_epochs 15 \
    --tune_epochs     15 \
    --lr_pretrain  2e-5 \
    --lr_tune      1e-4 \
    --bottleneck_dim 64 \
    2>&1 | tee results/training_logformer_v2.log