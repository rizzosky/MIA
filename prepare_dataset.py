"""
prepare_dataset.py
------------------
Etapa 1 del pipeline — corre UNA SOLA VEZ.

Lee los archivos JSON procesados, genera embeddings BERT para cada
evento, construye ventanas temporales deslizantes y guarda el
dataset completo en disco como windows.pkl.

Uso:
    python prepare_dataset.py \
        --normal_path    ../Data/Wazuh/processed/Legitimos \
        --incident_path  ../Data/Wazuh/processed/Incidente \
        --output         ./data/windows.pkl \
        --window_minutes 5 \
        --step_minutes   1 \
        --device         mps
"""

import os
import json
import glob
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de parseo
# ─────────────────────────────────────────────────────────────────────────────

def parse_timestamp(ts_str: str):
    """Parsea el timestamp de Wazuh eliminando el offset de zona horaria."""
    if not ts_str:
        return None
    t_pos = ts_str.find("T")
    if t_pos == -1:
        return None
    offset_pos = -1
    for i in range(len(ts_str) - 1, t_pos, -1):
        if ts_str[i] in ("+", "-"):
            offset_pos = i
            break
    ts_clean = ts_str[:offset_pos] if offset_pos != -1 else ts_str
    try:
        fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in ts_clean else "%Y-%m-%dT%H:%M:%S"
        return datetime.strptime(ts_clean, fmt)
    except ValueError:
        return None


def enrich_log_text(log: dict) -> str:
    """Construye el texto de entrada para BERT a partir de los campos del log."""
    parts = []
    if log.get("rule_id"):
        parts.append(f"Rule id: {log['rule_id']}")
    if log.get("rule_level"):
        parts.append(f"Rule level: {log['rule_level']}")
    if log.get("rule_firedtimes"):
        parts.append(f"Rule firedtimes: {log['rule_firedtimes']}")
    if log.get("process_id"):
        parts.append(f"process ID: {log['process_id']}")
    if log.get("thread_id"):
        parts.append(f"thread ID: {log['thread_id']}")
    return " | ".join(parts) if parts else "Wazuh log without specific details"


# ─────────────────────────────────────────────────────────────────────────────
# Carga de eventos desde JSONL
# ─────────────────────────────────────────────────────────────────────────────

def load_events_from_folder(folder: Path, is_incident: bool,
                            max_events: int = None) -> list:
    """
    Lee todos los archivos .json de una carpeta (JSONL, un evento por línea)
    y devuelve una lista de dicts con timestamp, texto enriquecido y label.

    Si max_events está definido y el número de eventos cargados lo supera,
    se aplica un submuestreo UNIFORME EN EL TIEMPO (índices equiespaciados
    sobre la secuencia ya ordenada cronológicamente), en lugar de truncar
    a los primeros N eventos. Esto preserva la cobertura temporal completa
    del período (inicio, mitad y fin), evitando perder, por ejemplo, la
    última parte de un período de varios días si el volumen es muy alto.
    """
    events = []
    files  = sorted(folder.glob("*.json"))
    print(f"  {'Incidente' if is_incident else 'Normal'}: {len(files)} archivo(s) en {folder}")

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
                    events.append({
                        "timestamp":   ts,
                        "text":        enrich_log_text(log),
                        "is_incident": is_incident,
                        "mitre_ids":   log.get("mitre_id") or [],
                    })
                except (json.JSONDecodeError, Exception):
                    continue

    events.sort(key=lambda e: e["timestamp"])
    n_original = len(events)

    if max_events is not None and n_original > max_events:
        idx = np.linspace(0, n_original - 1, max_events).astype(int)
        idx = sorted(set(idx))  # eliminar duplicados por redondeo
        events = [events[i] for i in idx]
        print(f"    → {n_original:,} eventos encontrados, "
              f"submuestreados uniformemente a {len(events):,} "
              f"(max_events={max_events:,})")
    else:
        print(f"    → {n_original:,} eventos cargados")

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Generación de embeddings en batch
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings_batch(texts: list, tokenizer, model, device: str,
                               batch_size: int = 64) -> np.ndarray:
    """
    Genera embeddings BERT (token CLS) para una lista de textos.
    Devuelve array de shape (n_texts, 768).
    """
    all_embeddings = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size),
                  desc="    Generando embeddings", leave=False):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embs)

        # Liberar memoria
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    return np.vstack(all_embeddings).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de ventanas temporales
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(events: list, embeddings: np.ndarray,
                  window_minutes: int, step_minutes: int) -> list:
    """
    Construye ventanas deslizantes sobre la secuencia de eventos.
    Cada ventana es un dict con:
        - embedding           : (768,)  promedio de los eventos en la ventana
        - embeddings_sequence : list de arrays (768,) — secuencia completa
        - has_anomaly         : 0 o 1
        - mitre_techniques    : set de técnicas MITRE en la ventana
        - n_events            : cantidad de eventos
        - start_time / end_time
    """
    if not events:
        return []

    window_len = timedelta(minutes=window_minutes)
    step_len   = timedelta(minutes=step_minutes)
    t_start    = events[0]["timestamp"]
    t_end      = events[-1]["timestamp"]

    # Índice rápido por timestamp
    timestamps = np.array([e["timestamp"].timestamp() for e in events])

    windows = []
    current = t_start

    while current + window_len <= t_end + step_len:
        win_start_ts = current.timestamp()
        win_end_ts   = (current + window_len).timestamp()

        idx = np.where((timestamps >= win_start_ts) &
                       (timestamps <= win_end_ts))[0]

        if len(idx) > 0:
            win_embs  = embeddings[idx]                        # (n, 768)
            avg_emb   = win_embs.mean(axis=0)                 # (768,)
            has_anom  = int(any(events[i]["is_incident"] for i in idx))
            mitre_set = set()
            for i in idx:
                ids = events[i]["mitre_ids"]
                if isinstance(ids, list):
                    mitre_set.update(ids)
                elif ids:
                    mitre_set.add(ids)

            windows.append({
                "embedding":            avg_emb,
                "embeddings_sequence":  [embeddings[i] for i in idx],
                "has_anomaly":          has_anom,
                "mitre_techniques":     list(mitre_set),
                "n_events":             len(idx),
                "start_time":           current,
                "end_time":             current + window_len,
            })

        current += step_len

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Split por origen (estratificado por clase)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split_by_origin(windows: list,
                                train_ratio: float = 0.70,
                                val_ratio:   float = 0.15,
                                seed:        int   = 42) -> tuple:
    """
    Divide las ventanas manteniendo la proporción de clases en cada split.

    Dado que los datos normales y los del incidente pertenecen a períodos
    temporales distintos y no comparables cronológicamente (el incidente
    ocurrió en marzo-abril 2025 y los datos normales en noviembre-diciembre
    2025), un split cronológico global concentraría todas las ventanas
    anómalas en un único split. En su lugar, se aplica un split estratificado
    por clase: se dividen por separado las ventanas normales y las anómalas,
    y luego se combinan. Esto garantiza que train, val y test contengan
    ventanas de ambas clases en proporciones similares.

    Este enfoque se documenta explícitamente en la tesis como decisión
    metodológica justificada por la naturaleza del dataset.
    """
    import random
    random.seed(seed)

    normales  = [w for w in windows if not w["has_anomaly"]]
    anomalas  = [w for w in windows if w["has_anomaly"]]

    # Ordenar cada grupo cronológicamente antes de dividir
    normales.sort(key=lambda w: w["start_time"])
    anomalas.sort(key=lambda w: w["start_time"])

    def split_group(group):
        n     = len(group)
        i_tr  = int(n * train_ratio)
        i_val = int(n * (train_ratio + val_ratio))
        return group[:i_tr], group[i_tr:i_val], group[i_val:]

    tr_n, val_n, te_n = split_group(normales)
    tr_a, val_a, te_a = split_group(anomalas)

    # Combinar y mezclar dentro de cada split
    def combine_and_shuffle(a, b):
        combined = a + b
        random.shuffle(combined)
        return combined

    train = combine_and_shuffle(tr_n, tr_a)
    val   = combine_and_shuffle(val_n, val_a)
    test  = combine_and_shuffle(te_n, te_a)

    print("\n── Split estratificado por origen ──")
    for name, ws in [("Train", train), ("Val", val), ("Test", test)]:
        pos = sum(1 for w in ws if w["has_anomaly"])
        neg = len(ws) - pos
        pct = pos / len(ws) * 100 if ws else 0
        print(f"  {name:<6}: {len(ws):>5} ventanas  "
              f"(anómalas={pos:>4} [{pct:.1f}%], normales={neg:>5})")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time
    parser = argparse.ArgumentParser(description="Preprocesamiento de logs SOC")
    parser.add_argument("--normal_path",    type=Path, required=True)
    parser.add_argument("--incident_path",  type=Path, required=True)
    parser.add_argument("--output",         type=Path, default=Path("./data/windows.pkl"))
    parser.add_argument("--window_minutes", type=int,  default=5)
    parser.add_argument("--step_minutes",   type=int,  default=1)
    parser.add_argument("--bert_model",     type=str,  default="bert-base-uncased")
    parser.add_argument("--batch_size",     type=int,  default=64)
    parser.add_argument("--device",         type=str,  default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max_events_per_class", type=int, default=None,
                        help="Máximo de eventos a cargar por clase "
                            "(normal/incidente). Si se supera, se aplica "
                            "submuestreo uniforme en el tiempo. Útil para "
                            "sistemas con volumen muy alto (p.ej. pfSense).")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t_pipeline_start = time.time()

    print(f"\n{'='*60}")
    print("  ETAPA 1 — PREPROCESAMIENTO")
    print(f"{'='*60}")
    print(f"  Normal:   {args.normal_path}")
    print(f"  Incidente:{args.incident_path}")
    print(f"  Ventana:  {args.window_minutes} min  |  Paso: {args.step_minutes} min")
    print(f"  Salida:   {args.output}")

    # 1. Cargar eventos
    t0 = time.time()
    print("\n[1/4] Cargando eventos...")
    normal_events   = load_events_from_folder(
        args.normal_path,   is_incident=False,
        max_events=args.max_events_per_class)
    incident_events = load_events_from_folder(
        args.incident_path, is_incident=True,
        max_events=args.max_events_per_class)
    all_events      = sorted(normal_events + incident_events,
                             key=lambda e: e["timestamp"])
    t_load = time.time() - t0
    print(f"  Total eventos: {len(all_events):,}")
    print(f"  Tiempo de carga: {t_load:.1f}s")

    # 2. Generar embeddings
    t0 = time.time()
    print("\n[2/4] Cargando modelo BERT...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    bert      = BertModel.from_pretrained(args.bert_model).to(args.device)
    t_load_bert = time.time() - t0

    t0 = time.time()
    print("[2/4] Generando embeddings...")
    texts      = [e["text"] for e in all_events]
    embeddings = generate_embeddings_batch(
        texts, tokenizer, bert, args.device, args.batch_size
    )
    t_embeddings = time.time() - t0
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Tiempo de carga de BERT: {t_load_bert:.1f}s")
    print(f"  Tiempo de generación de embeddings: {t_embeddings:.1f}s "
          f"({len(all_events)/max(t_embeddings,1e-6):.1f} eventos/s)")

    # Liberar BERT de memoria (ya no se necesita)
    del bert
    if args.device == "mps":
        torch.mps.empty_cache()
    elif args.device == "cuda":
        torch.cuda.empty_cache()

    # 3. Construir ventanas
    t0 = time.time()
    print("\n[3/4] Construyendo ventanas temporales...")
    windows = build_windows(
        all_events, embeddings,
        args.window_minutes, args.step_minutes
    )
    t_windows = time.time() - t0
    n_anom = sum(1 for w in windows if w["has_anomaly"])
    n_norm = len(windows) - n_anom
    print(f"  Total ventanas: {len(windows):,}  "
          f"(anómalas={n_anom:,}, normales={n_norm:,})")
    print(f"  Tiempo de construcción de ventanas: {t_windows:.1f}s")

    # 4. Split cronológico y guardar
    t0 = time.time()
    print("\n[4/4] Aplicando split cronológico y guardando...")
    train, val, test = stratified_split_by_origin(windows)

    t_pipeline_total = time.time() - t_pipeline_start

    dataset = {
        "train":                train,
        "val":                  val,
        "test":                 test,
        "all":                  windows,
        "window_minutes":       args.window_minutes,
        "step_minutes":         args.step_minutes,
        "bert_model":           args.bert_model,
        "embedding_dim":        embeddings.shape[1],
        "max_events_per_class": args.max_events_per_class,
        "created_at":           datetime.now().isoformat(),
        "timing": {
            "n_events_total":         len(all_events),
            "load_events_s":          round(t_load, 2),
            "load_bert_s":            round(t_load_bert, 2),
            "generate_embeddings_s":  round(t_embeddings, 2),
            "build_windows_s":        round(t_windows, 2),
            "total_pipeline_s":       round(t_pipeline_total, 2),
            "events_per_second":      round(len(all_events)/max(t_embeddings,1e-6), 2),
        },
    }

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\n  Dataset guardado en {args.output} ({size_mb:.1f} MB)")
    print(f"\n{'='*60}")
    print("  RESUMEN DE TIEMPOS")
    print(f"{'='*60}")
    print(f"  Carga de eventos:          {t_load:.1f}s")
    print(f"  Carga de BERT:             {t_load_bert:.1f}s")
    print(f"  Generación de embeddings:  {t_embeddings:.1f}s "
          f"({len(all_events)/max(t_embeddings,1e-6):.1f} eventos/s)")
    print(f"  Construcción de ventanas:  {t_windows:.1f}s")
    print(f"  TIEMPO TOTAL DEL PIPELINE: {t_pipeline_total:.1f}s "
          f"({t_pipeline_total/60:.1f} min)")
    print(f"{'='*60}")
    print("  Preprocesamiento completado.")
    print(f"  Para entrenar los modelos:")
    print(f"  python run_experiments.py --dataset {args.output} --device {args.device}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()