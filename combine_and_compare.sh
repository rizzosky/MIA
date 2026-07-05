#!/bin/bash

# Combinar ambos logs en uno solo
cat results/training_v2.log results/training_logformer_v2.log > results/training_full_v2.log

# Parsear el log combinado
python parse_training_log.py \
    --log        results/training_full_v2.log \
    --output_dir results/curves