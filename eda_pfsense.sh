#!/bin/bash

ORGANIZACION="ORGANIZACION"
python eda_soc_logs.py \
    --normal ../Data/Wazuh/processed/Legitimos/pfsense_${ORGANIZACION} --incidente ../Data/Wazuh/processed/Incidentes/pfsense_${ORGANIZACION} \
    --salida ./eda_output/pfsense/