#!/bin/bash
python prepare_dataset_ruleid.py \
    --normal_path   ../Data/Wazuh/processed/Legitimos/task_scheduler_ASESP \
    --incident_path ../Data/Wazuh/processed/Incidentes/task_scheduler_ASESP \
    --output        ./data/windows.pkl