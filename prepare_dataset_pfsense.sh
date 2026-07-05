#!/bin/bash

python prepare_dataset.py \
    --normal_path   ../Data/Wazuh/processed/Legitimos/pfsense_ASESP \
    --incident_path ../Data/Wazuh/processed/Incidentes/pfsense_ASESP \
    --output        ./data/windows_pfsense_500k.pkl \
    --window_minutes 5 \
    --step_minutes   1 \
    --device         mps \
    --max_events_per_class 500000