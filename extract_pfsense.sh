#!/bin/bash
# extract_pfsense.sh
# -------------------
# Extrae los eventos del agente 016 (pfSense) combinando las dos
# locations disponibles (rsyslog y pfBlocker), tanto para el
# período normal como para el incidente. Reutiliza exactamente
# el mismo criterio de filtrado que el experimento original
# (Sección 3.2 del capítulo de Desarrollo), cambiando únicamente
# agent_id y permitiendo dos locations en lugar de una.
#
# Uso:
#   ./extract_pfsense.sh
#
# Ajustar las rutas ORGANIZACION, RAW_NORMAL, RAW_INCIDENTE,
# OUT_NORMAL, OUT_INCIDENTE según tu estructura de carpetas.

set -e

ORGANIZACION="ORGANIZACION"
AGENT_ID="016"

RAW_NORMAL="normales"
RAW_INCIDENTE="incidentes"

OUT_NORMAL="../processed/Legitimos/pfsense_${ORGANIZACION}"
OUT_INCIDENTE="../processed/Incidentes/pfsense_${ORGANIZACION}"

mkdir -p "$OUT_NORMAL" "$OUT_INCIDENTE"

echo "=============================================="
echo "  Extrayendo agente ${AGENT_ID} (pfSense)"
echo "  Locations: rsyslog + pfBlocker"
echo "=============================================="

echo ""
echo "[1/2] Procesando período NORMAL..."
find "$RAW_NORMAL" -name "*.xz" | \
parallel -j 8 "xzcat {} | jq -c '
  select(
    .agent.labels.Group == \"${ORGANIZACION}\" and
    .agent.id == \"${AGENT_ID}\" and
    (.location == \"/var/log/rsyslog/received.log\" or
     .location == \"/var/log/rsyslog/PfBlocker_log/pfblocker.log\")
  ) | {
    timestamp:       .timestamp,
    rule_id:         .rule.id,
    rule_level:      .rule.level,
    rule_firedtimes: .rule.firedtimes,
    mitre_id:        (.rule.mitre.id // null),
    mitre_tactic:    (.rule.mitre.tactic // null),
    location:        .location,
  }' > ${OUT_NORMAL}/\$(basename {} .xz)_pfsense_filtrado.json"

echo "[2/2] Procesando período INCIDENTE..."
find "$RAW_INCIDENTE" -name "*.xz" | \
parallel -j 8 "xzcat {} | jq -c '
  select(
    .agent.labels.Group == \"${ORGANIZACION}\" and
    .agent.id == \"${AGENT_ID}\" and
    (.location == \"/var/log/rsyslog/received.log\" or
     .location == \"/var/log/rsyslog/PfBlocker_log/pfblocker.log\")
  ) | {
    timestamp:       .timestamp,
    rule_id:         .rule.id,
    rule_level:      .rule.level,
    rule_firedtimes: .rule.firedtimes,
    mitre_id:        (.rule.mitre.id // null),
    mitre_tactic:    (.rule.mitre.tactic // null),
    location:        .location,
  }' > ${OUT_INCIDENTE}/\$(basename {} .xz)_pfsense_filtrado.json"

echo ""
echo "=============================================="
echo "  Extracción completa."
echo "  Normal:    $OUT_NORMAL"
echo "  Incidente: $OUT_INCIDENTE"
echo "=============================================="

echo ""
echo "Conteo de eventos extraídos:"
echo "  Normal:    $(cat ${OUT_NORMAL}/*.json 2>/dev/null | wc -l)"
echo "  Incidente: $(cat ${OUT_INCIDENTE}/*.json 2>/dev/null | wc -l)"