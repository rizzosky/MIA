#!/bin/bash

# Combinar ambos logs en uno solo
cat results_pfsense/training_pfsense.log results_pfsense/training_logformer_pfsense.log > results_pfsense/training_full_pfsense.log

# Parsear el log combinado
python parse_training_log.py \
    --log        results_pfsense/training_full_pfsense.log \
    --output_dir results_pfsense/curves