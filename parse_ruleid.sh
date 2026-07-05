#!/bin/bash

# Combinar los dos logs de rule_id en uno
cat results_ruleid/training_ruleid.log \
    results_ruleid/training_logformer_ruleid.log \
    > results_ruleid/training_full_ruleid.log

# Parsear y generar figuras
python parse_ruleid.py \
    --log_path   results_ruleid/training_full_ruleid.log \
    --output_dir results_ruleid/curves

python plot_ruleid.py \
    --results_dir       ./results \
    --results_ruleid_dir ./results_ruleid \
    --output_dir        ./results/curves