#!/bin/bash

ORGANIZACION="ORGANIZACION"

python eda_soc_logs.py \
    --normal ../Data/Wazuh/processed/Legitimos/task_scheduler_${ORGANIZACION} --incidente ../Data/Wazuh/processed/Incidentes/task_scheduler_${ORGANIZACION} \
    --salida ./eda_output/task_scheduler/