#!/bin/bash

ORGANIZACION="ORGANIZACION"
python prepare_dataset_ruleid.py \
    --normal_path   ../Data/Wazuh/processed/Legitimos/task_scheduler_${ORGANIZACION} \
    --incident_path ../Data/Wazuh/processed/Incidentes/task_scheduler_${ORGANIZACION} \
    --output        ./data/windows.pkl