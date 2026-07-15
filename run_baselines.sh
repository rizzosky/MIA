#!/bin/bash

mkdir -p results_baselines results_baselines_ruleid results_baselines_ruleid_tfidf
# 1. Baselines con embedding BERT (comparación a igualdad de representación
#    con mis redes neuronales)
python run_baselines.py \
    --dataset    ./data/windows.pkl \
    --output_dir ./results_baselines \
    2>&1 | tee results_baselines/training_baselines.log

# 2. Baselines con vector de conteo (fiel a Xu et al. 2009, la forma
#    original en que se usan estos métodos en la literatura)
python run_baselines_ruleid.py \
    --dataset    ./data/windows_ruleid.pkl \
    --output_dir ./results_baselines_ruleid \
    2>&1 | tee results_baselines_ruleid/training_baselines_ruleid.log

# 2b. Variante con TF-IDF (la mejora que reporta el propio paper de Xu et al.)
python run_baselines_ruleid.py \
    --dataset    ./data/windows_ruleid.pkl \
    --output_dir ./results_baselines_ruleid_tfidf \
    --tfidf \
    2>&1 | tee results_baselines_ruleid_tfidf/training_baselines_ruleid_tfidf.log