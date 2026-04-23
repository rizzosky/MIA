import os
import json
import glob
import pickle
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch


class TimeBasedLogProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(config.device)
        self.bert_model.eval()

    # --------------------------------------------------------------
    # 1. Parseo de timestamp (formato Wazuh)
    # --------------------------------------------------------------
    def extract_timestamp(self, log):
        """Parsea timestamp de Wazuh: '2023-10-16T12:12:18.684-0000'"""
        ts_str = log.get('timestamp', '')
        if not ts_str:
            return None
        # Eliminar offset (ej: -0000, +0000) que puede ir después de la T
        # Buscar el último '+' o '-' después de la T
        t_pos = ts_str.find('T')
        if t_pos == -1:
            return None
        # Recorrer desde el final hasta encontrar un '+' o '-'
        offset_pos = -1
        for i in range(len(ts_str)-1, t_pos, -1):
            if ts_str[i] in ('+', '-'):
                offset_pos = i
                break
        if offset_pos != -1:
            ts_clean = ts_str[:offset_pos]
        else:
            ts_clean = ts_str
        # Parsear
        try:
            if '.' in ts_clean:
                return datetime.strptime(ts_clean, '%Y-%m-%dT%H:%M:%S.%f')
            else:
                return datetime.strptime(ts_clean, '%Y-%m-%dT%H:%M:%S')
        except:
            return None

    # --------------------------------------------------------------
    # 2. Extraer información enriquecida de cada log
    # --------------------------------------------------------------
    def enrich_log_text(self, log):
        """Construye un string descriptivo a partir de los campos del log."""
        parts = []

        # Regla y nivel
        if log.get('rule_id'):
            parts.append(f"Rule id: {log['rule_id']}")
        if log.get('rule_level'):
            parts.append(f"Rule level: {log['rule_level']}")
        if log.get('rule_firedtimes'):
            parts.append(f"Rule firedtimes: {log['rule_firedtimes']}")

        if log.get('process_id'):
            parts.append(f"process ID: {log['process_id']}")
        if log.get('thread_id'):
            parts.append(f"thread ID:{log['thread_id']}")

        # Si no hay nada, usar descripción de la regla o un placeholder
        if not parts:
            return "Wazuh log without specific details"

        return " | ".join(parts)

    # --------------------------------------------------------------
    # 3. Generar embedding para un texto (con caché opcional)
    # --------------------------------------------------------------
    def generate_embedding(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.config.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Usar el token [CLS]
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return emb

    # --------------------------------------------------------------
    # 4. Procesar un chunk de tiempo (archivos que caen en un rango)
    # --------------------------------------------------------------
    def process_time_chunk_by_timestamps(self, chunk_info, chunk_id, max_events=None):
        print(f"\n{'='*60}")
        print(f"PROCESANDO CHUNK: {chunk_id}")
        print(f"Rango: {chunk_info['start_time']} -> {chunk_info['end_time']}")
        print(f"Archivos: {len(chunk_info['files'])}")
        print(f"{'='*60}")

        events = []  # lista de dicts con timestamp, embedding, has_mitre, mitre_ids
        total_events = 0

        for file_info in chunk_info['files']:
            filepath = file_info['path']
            is_incident_file = file_info.get('is_incident', False)

            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        log = json.loads(line)
                        ts = self.extract_timestamp(log)
                        if ts is None:
                            continue
                        if ts < chunk_info['start_time'] or ts > chunk_info['end_time']:
                            continue

                        # Texto enriquecido
                        enriched = self.enrich_log_text(log)
                        emb = self.generate_embedding(enriched)

                        # MITRE
                        mitre_info = log.get('rule', {}).get('mitre', {})
                        mitre_ids = mitre_info.get('id', [])
                        has_mitre = len(mitre_ids) > 0

                        events.append({
                            'timestamp': ts,
                            'embedding': emb,
                            'has_mitre': has_mitre,
                            'mitre_ids': mitre_ids,
                            'is_incident_file': is_incident_file
                        })
                        total_events += 1
                        if max_events and total_events >= max_events:
                            break
                    except:
                        continue
            if max_events and total_events >= max_events:
                break

        if not events:
            print("  No se encontraron eventos en este chunk.")
            return []

        # Ordenar por timestamp
        events.sort(key=lambda x: x['timestamp'])

        # Crear ventanas deslizantes basadas en tiempo (5 min, stride 1 min)
        windows = self.create_time_windows(events, chunk_info['start_time'], chunk_info['end_time'])
        print(f"  Eventos en chunk: {len(events)}")
        print(f"  Ventanas creadas: {len(windows)}")
        return windows

    # --------------------------------------------------------------
    # 5. Crear ventanas de tiempo con stride
    # --------------------------------------------------------------
    def create_time_windows(self, events, chunk_start, chunk_end):
        window_len = timedelta(minutes=self.config.window_minutes)
        step_len = timedelta(minutes=self.config.step_minutes)

        windows = []
        current = chunk_start
        # Avanzar hasta cubrir todo el chunk
        while current + window_len <= chunk_end:
            win_start = current
            win_end = current + window_len

            # Seleccionar eventos dentro de esta ventana
            win_events = [e for e in events if win_start <= e['timestamp'] <= win_end]

            if win_events:
                # Embedding promedio de la ventana
                embeddings = np.array([e['embedding'] for e in win_events])
                avg_emb = np.mean(embeddings, axis=0).astype(np.float16)
                has_anomaly = any(e['is_incident_file'] for e in win_events)
                # Recopilar técnicas MITRE únicas
                mitre_set = set()
                for e in win_events:
                    mitre_set.update(e['mitre_ids'])
                windows.append({
                    'embedding': avg_emb,
                    'embeddings_sequence': [e['embedding'] for e in win_events],  # para modelo secuencial
                    'has_anomaly': int(has_anomaly),
                    'mitre_techniques': list(mitre_set),
                    'n_events': len(win_events),
                    'start_time': win_start,
                    'end_time': win_end,
                })
            current += step_len

        return windows

    # --------------------------------------------------------------
    # 6. Métodos para manejo de chunks (escaneo, guardado)
    # --------------------------------------------------------------
    def group_files_by_time_chunks(self, normal_paths, incident_paths, chunk_minutes=360):
        """Escanea todos los archivos, extrae rangos de tiempo y los agrupa en chunks."""
        print("Escaneando archivos para determinar rangos de tiempo...")
        all_files = []
        for p in normal_paths:
            all_files.extend(glob.glob(f"{p}/**/*.json", recursive=True))
        for p in incident_paths:
            all_files.extend(glob.glob(f"{p}/**/*.json", recursive=True))

        # Obtener min y max timestamp de cada archivo (muestreo)
        file_ranges = {}
        for fpath in tqdm(all_files, desc="Leyendo timestamps"):
            timestamps = []
            with open(fpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        log = json.loads(line)
                        ts = self.extract_timestamp(log)
                        if ts:
                            timestamps.append(ts)
                            if len(timestamps) >= 1000:  # muestra suficiente
                                break
                    except:
                        continue
            if timestamps:
                file_ranges[fpath] = {
                    'min_time': min(timestamps),
                    'max_time': max(timestamps),
                    'is_incident': any(p in fpath for p in incident_paths)
                }

        if not file_ranges:
            raise ValueError("No se pudo extraer ningún timestamp de los archivos.")

        global_min = min(v['min_time'] for v in file_ranges.values())
        global_max = max(v['max_time'] for v in file_ranges.values())
        print(f"Rango temporal global: {global_min} -> {global_max}")

        # Crear chunks
        chunks = {}
        chunk_delta = timedelta(minutes=chunk_minutes)
        start = global_min
        chunk_id = 0
        while start < global_max:
            end = start + chunk_delta
            key = f"chunk_{chunk_id:04d}_{start.strftime('%Y%m%d_%H%M')}"
            chunks[key] = {
                'start_time': start,
                'end_time': end,
                'files': [],
                'has_incidents': False
            }
            start = end
            chunk_id += 1

        # Asignar archivos a chunks
        for fpath, info in file_ranges.items():
            f_min = info['min_time']
            f_max = info['max_time']
            for ckey, cinfo in chunks.items():
                if f_min <= cinfo['end_time'] and f_max >= cinfo['start_time']:
                    cinfo['files'].append({
                        'path': fpath,
                        'is_incident': info['is_incident']
                    })
                    if info['is_incident']:
                        cinfo['has_incidents'] = True

        # Eliminar chunks sin archivos
        chunks = {k: v for k, v in chunks.items() if v['files']}
        return chunks

    def save_chunk_metadata(self, chunks, filename='chunk_metadata.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(chunks, f)

    def load_chunk_metadata(self, filename='chunk_metadata.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None