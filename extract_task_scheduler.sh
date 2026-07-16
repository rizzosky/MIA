#!/bin/bash
# extract_task_scheduler.sh
# -------------------
# Extrae los eventos del agente 256 (Windows Task Scheduler,
# location EventChannel), tanto para el período normal como para
# el incidente. Aplica el criterio de filtrado y los campos
# documentados en la Sección de Recolección y filtrado de logs
# del capítulo de Metodología.
#
# Uso:
#   ./extract_task_scheduler.sh
#
# Ajustar las rutas ORGANIZACION, RAW_NORMAL, RAW_INCIDENTE,
# OUT_NORMAL, OUT_INCIDENTE según tu estructura de carpetas.

set -e

ORGANIZACION="ORGANIZACION"
AGENT_ID="256"

RAW_NORMAL="normales"
RAW_INCIDENTE="incidentes"

OUT_NORMAL="../processed/Legitimos/task_scheduler_${ORGANIZACION}"
OUT_INCIDENTE="../processed/Incidentes/task_scheduler_${ORGANIZACION}"

mkdir -p "$OUT_NORMAL" "$OUT_INCIDENTE"

echo "=============================================="
echo "  Extrayendo agente ${AGENT_ID} (Windows Task Scheduler) de la organización ${ORGANIZACION}"
echo "  Location: EventChannel"
echo "=============================================="

echo ""
echo "[1/2] Procesando período NORMAL..."
find "$RAW_NORMAL" -name "*.xz" | \
parallel -j 8 "xzcat {} | jq -c '
  select(
    .agent.labels.Group == \"${ORGANIZACION}\" and
    .agent.id == \"${AGENT_ID}\" and
    .location == \"EventChannel\"
  ) | {
    timestamp:       .timestamp,
    rule_id:         .rule.id,
    rule_level:      .rule.level,
    rule_firedtimes: .rule.firedtimes,
    mitre_id:        (.rule.mitre.id // null),
    mitre_tactic:    (.rule.mitre.tactic // null),
    agent_id:        .agent.id,
    location:        .location,
    process_id:      .data.win.system.processID,
    thread_id:       .data.win.system.threadID
  }' > ${OUT_NORMAL}/\$(basename {} .xz)_task_scheduler_filtrado.json"

echo "[2/2] Procesando período INCIDENTE..."
find "$RAW_INCIDENTE" -name "*.xz" | \
parallel -j 8 "xzcat {} | jq -c '
  select(
    .agent.labels.Group == \"${ORGANIZACION}\" and
    .agent.id == \"${AGENT_ID}\" and
    .location == \"EventChannel\"
  ) | {
    timestamp:       .timestamp,
    rule_id:         .rule.id,
    rule_level:      .rule.level,
    rule_firedtimes: .rule.firedtimes,
    mitre_id:        (.rule.mitre.id // null),
    mitre_tactic:    (.rule.mitre.tactic // null),
    agent_id:        .agent.id,
    location:        .location,
    process_id:      .data.win.system.processID,
    thread_id:       .data.win.system.threadID
  }' > ${OUT_INCIDENTE}/\$(basename {} .xz)_task_scheduler_filtrado.json"

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