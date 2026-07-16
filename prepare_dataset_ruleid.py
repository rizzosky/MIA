"""
prepare_dataset_ruleid.py
--------------------------
Variante de prepare_dataset.py para el experimento "log key only":
en lugar de generar embeddings BERT a partir de texto enriquecido,
cada evento se representa únicamente por su rule_id, mapeado a un
índice categórico. Esto replica el esquema clásico de DeepLog,
LogAnomaly y LogBERT, donde cada "log key" es un identificador de
template (similar al event_id de HDFS/BGL/OpenStack), y la
representación vectorial se aprende desde cero mediante una capa
de embedding entrenable (nn.Embedding), no mediante un encoder
de lenguaje preentrenado.

Esto permite comparar dos paradigmas de representación bajo las
mismas condiciones de ventana y etiquetado:
    1. Texto enriquecido + BERT preentrenado (prepare_dataset.py)
    2. Identificador categórico + embedding aprendido (este script)

Uso:
    python prepare_dataset_ruleid.py \
        --normal_path    ../Data/Wazuh/processed/Legitimos/task_scheduler_${ORGANIZACION} \
        --incident_path  ../Data/Wazuh/processed/Incidentes/task_scheduler_${ORGANIZACION} \
        --output         ./data/windows_ruleid.pkl \
        --window_minutes 5 \
        --step_minutes   1
"""

import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from prepare_dataset import parse_timestamp, stratified_split_by_origin


# ─────────────────────────────────────────────────────────────────────────────
# Carga de eventos (solo rule_id + timestamp + label)
# ─────────────────────────────────────────────────────────────────────────────

def load_events_ruleid(folder: Path, is_incident: bool) -> list:
    events = []
    files  = sorted(folder.glob("*.json"))
    print(f"  {'Incidente' if is_incident else 'Normal'}: "
          f"{len(files)} archivo(s) en {folder}")

    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log = json.loads(line)
                    ts  = parse_timestamp(log.get("timestamp", ""))
                    if ts is None:
                        continue
                    rule_id = log.get("rule_id")
                    if rule_id is None:
                        continue
                    events.append({
                        "timestamp":   ts,
                        "rule_id":     str(rule_id),
                        "is_incident": is_incident,
                        "mitre_ids":   log.get("mitre_id") or [],
                    })
                except (json.JSONDecodeError, Exception):
                    continue

    events.sort(key=lambda e: e["timestamp"])
    print(f"    → {len(events):,} eventos cargados")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulario de rule_id
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(events: list, train_cutoff_idx: int) -> dict:
    """
    Construye el vocabulario de rule_id ÚNICAMENTE a partir de los
    eventos de entrenamiento, para evitar leakage del vocabulario
    de validación/test hacia el entrenamiento. Los rule_id no vistos
    durante el entrenamiento se mapean al índice especial <UNK>.

    Índice 0 reservado para <PAD>, índice 1 para <UNK>.
    """
    train_events = events[:train_cutoff_idx]
    unique_ids = sorted(set(e["rule_id"] for e in train_events))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, rid in enumerate(unique_ids, start=2):
        vocab[rid] = i

    print(f"  Vocabulario construido: {len(vocab)} tokens "
          f"({len(unique_ids)} rule_id distintos + PAD + UNK)")
    return vocab


def encode_rule_id(rule_id: str, vocab: dict) -> int:
    return vocab.get(rule_id, vocab["<UNK>"])


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de ventanas (idéntico criterio temporal que prepare_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_windows_ruleid(events: list, vocab: dict,
                         window_minutes: int, step_minutes: int) -> list:
    if not events:
        return []

    window_len = timedelta(minutes=window_minutes)
    step_len   = timedelta(minutes=step_minutes)
    t_start    = events[0]["timestamp"]
    t_end      = events[-1]["timestamp"]

    timestamps = np.array([e["timestamp"].timestamp() for e in events])

    windows = []
    current = t_start

    while current + window_len <= t_end + step_len:
        win_start_ts = current.timestamp()
        win_end_ts   = (current + window_len).timestamp()

        idx = np.where((timestamps >= win_start_ts) &
                       (timestamps <= win_end_ts))[0]

        if len(idx) > 0:
            rule_id_seq = [encode_rule_id(events[i]["rule_id"], vocab)
                           for i in idx]
            has_anom = int(any(events[i]["is_incident"] for i in idx))
            mitre_set = set()
            for i in idx:
                ids = events[i]["mitre_ids"]
                if isinstance(ids, list):
                    mitre_set.update(ids)
                elif ids:
                    mitre_set.add(ids)

            windows.append({
                "rule_id_sequence": rule_id_seq,   # lista de enteros
                "has_anomaly":      has_anom,
                "mitre_techniques": list(mitre_set),
                "n_events":         len(idx),
                "start_time":       current,
                "end_time":         current + window_len,
            })

        current += step_len

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_path",    type=Path, required=True)
    parser.add_argument("--incident_path",  type=Path, required=True)
    parser.add_argument("--output",         type=Path,
                        default=Path("./data/windows_ruleid.pkl"))
    parser.add_argument("--window_minutes", type=int, default=5)
    parser.add_argument("--step_minutes",   type=int, default=1)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  PREPROCESAMIENTO — Esquema log-key (rule_id only)")
    print(f"{'='*60}")

    print("\n[1/4] Cargando eventos...")
    normal_events   = load_events_ruleid(args.normal_path,   is_incident=False)
    incident_events = load_events_ruleid(args.incident_path, is_incident=True)
    all_events      = sorted(normal_events + incident_events,
                             key=lambda e: e["timestamp"])
    print(f"  Total eventos: {len(all_events):,}")

    # Para construir el vocabulario sin leakage, se separan primero
    # los eventos por clase y se ordenan cronológicamente, igual
    # que en el split de prepare_dataset.py, y se usa solo la
    # porción de "entrenamiento" (70% inicial de cada clase) para
    # construir el vocabulario.
    print("\n[2/4] Construyendo vocabulario de rule_id...")
    normales_sorted  = sorted([e for e in all_events if not e["is_incident"]],
                              key=lambda e: e["timestamp"])
    anomalas_sorted  = sorted([e for e in all_events if e["is_incident"]],
                              key=lambda e: e["timestamp"])
    train_events_for_vocab = (
        normales_sorted[: int(len(normales_sorted) * 0.70)] +
        anomalas_sorted[: int(len(anomalas_sorted) * 0.70)]
    )
    vocab = build_vocab(all_events, len(train_events_for_vocab))
    # Nota: build_vocab espera una lista con el cutoff por índice;
    # como ya separamos antes, construimos el vocab directamente:
    unique_ids = sorted(set(e["rule_id"] for e in train_events_for_vocab))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, rid in enumerate(unique_ids, start=2):
        vocab[rid] = i
    print(f"  Vocabulario final: {len(vocab)} tokens")

    print("\n[3/4] Construyendo ventanas temporales...")
    windows = build_windows_ruleid(
        all_events, vocab, args.window_minutes, args.step_minutes
    )
    n_anom = sum(1 for w in windows if w["has_anomaly"])
    n_norm = len(windows) - n_anom
    print(f"  Total ventanas: {len(windows):,}  "
          f"(anómalas={n_anom:,}, normales={n_norm:,})")

    print("\n[4/4] Aplicando split estratificado por origen y guardando...")
    train, val, test = stratified_split_by_origin(windows)

    dataset = {
        "train":          train,
        "val":            val,
        "test":           test,
        "all":            windows,
        "vocab":          vocab,
        "vocab_size":     len(vocab),
        "window_minutes": args.window_minutes,
        "step_minutes":   args.step_minutes,
        "representation": "rule_id_only",
        "created_at":     datetime.now().isoformat(),
    }

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\n  Dataset guardado en {args.output} ({size_mb:.2f} MB)")
    print(f"  Vocabulario: {len(vocab)} tokens (incluye <PAD> y <UNK>)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()