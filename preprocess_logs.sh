#!/bin/bash

cd incidentes

find . -name "filename_incidentes.xz" | \
parallel -j 7 'xzcat {} | jq -c "select(.agent.labels.Group == \"agent_organization\" and .agent.id == \"256\" and .location == \"EventChannel\") |
  {
    timestamp: .timestamp,
    rule_id: .rule.id,
    rule_level: .rule.level,
    rule_firedtimes: .rule.firedtimes,
    mitre_id: (.rule.mitre.id // null),
    mitre_tactic: (.rule.mitre.tactic // null),
    agent_id: .agent.id,
    location: .location,
    process_id: .data.win.system.processID,
    thread_id: .data.win.system.threadID
  }" > ../../processed/Incidente/$(basename {} .xz)_filtrado.json'


cd ../normales

find . -name "filename_normales.xz" | \
parallel -j 7 'xzcat {} | jq -c "select(.agent.labels.Group == \"agent_organization\" and .agent.id == \"256\" and .location == \"EventChannel\") |
  {
    timestamp: .timestamp,
    rule_id: .rule.id,
    rule_level: .rule.level,
    rule_firedtimes: .rule.firedtimes,
    mitre_id: (.rule.mitre.id // null),
    mitre_tactic: (.rule.mitre.tactic // null),
    agent_id: .agent.id,
    location: .location,
    process_id: .data.win.system.processID,
    thread_id: .data.win.system.threadID
  }" > ../../processed/Legitimos/$(basename {} .xz)_filtrado.json'

