#!/bin/bash
ORGANIZACION="ORGANIZACION"

python prepare_dataset.py \
    --normal_path   ../Data/Wazuh/processed/Legitimos/pfsense_${ORGANIZACION} \
    --incident_path ../Data/Wazuh/processed/Incidentes/pfsense_${ORGANIZACION} \
    --output        ./data/windows_pfsense.pkl \
    --window_minutes 5 \
    --step_minutes   1 \
    --device         mps \
    --max_events_per_class 50000 \
    2>&1 | tee ./data/prepare_dataset_pfsense_50k.log