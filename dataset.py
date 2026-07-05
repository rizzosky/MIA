"""
dataset.py
----------
Dataset y collate_fn para ventanas temporales con embeddings BERT
precalculados (secuencia completa o promedio).

CAMBIO IMPORTANTE respecto a la versión original:
El recorte de secuencias más largas que max_seq_len ya NO toma los
primeros N eventos cronológicamente. En su lugar, aplica un
MUESTREO UNIFORME EN EL TIEMPO sobre los índices de la secuencia,
preservando eventos representativos de todo el intervalo de la
ventana (inicio, mitad y fin) en lugar de sesgar sistemáticamente
hacia el comienzo.

Esto es especialmente relevante para sistemas de alto volumen
(p.ej. firewalls) donde una ventana puede contener cientos o miles
de eventos: tomar los primeros N descarta casi toda la ventana de
forma no representativa, mientras que el muestreo uniforme
mantiene una "silueta" temporal de la actividad completa.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


def _uniform_subsample_indices(n_total: int, n_target: int) -> np.ndarray:
    """
    Devuelve n_target índices equiespaciados en [0, n_total-1],
    preservando el primer y último elemento. Si n_total <= n_target,
    devuelve todos los índices sin modificar.
    """
    if n_total <= n_target:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, n_target).astype(int)


class TimeWindowDataset(Dataset):
    def __init__(self, windows, use_sequence=False, max_seq_len=100):
        self.windows = windows
        self.use_sequence = use_sequence
        if use_sequence and max_seq_len is None:
            lengths = [len(w['embeddings_sequence']) for w in windows]
            self.max_seq_len = int(np.percentile(lengths, 95)) if lengths else 1
        else:
            self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.use_sequence:
            seq_list = window['embeddings_sequence']
            n_events = len(seq_list)

            if n_events > self.max_seq_len:
                # Muestreo uniforme en el tiempo, NO los primeros N.
                sample_idx = _uniform_subsample_indices(n_events, self.max_seq_len)
                seq_list = [seq_list[i] for i in sample_idx]

            seq = np.array(seq_list)
            seq = torch.FloatTensor(seq)

            if seq.shape[0] < self.max_seq_len:
                padding = torch.zeros(self.max_seq_len - seq.shape[0], seq.shape[1])
                seq = torch.cat([seq, padding], dim=0)

            embeddings = seq
            label = torch.tensor([window['has_anomaly']], dtype=torch.float32)
        else:
            # Siempre usar embedding promedio (evita problema de longitud variable)
            embeddings = torch.FloatTensor(window['embedding'])
            label = torch.tensor([window['has_anomaly']], dtype=torch.float32)

        meta = {
            'has_anomaly':      window['has_anomaly'],
            'mitre_techniques': window.get('mitre_techniques', []),
            'n_events':         window.get('n_events', 0),
            'avg_embedding':    window.get('embedding', []).tolist() if 'embedding' in window else None,
            'start_time':       window.get('start_time').isoformat() if 'start_time' in window else None,
            'end_time':         window.get('end_time').isoformat() if 'end_time' in window else None,
        }

        return embeddings, label, meta


def collate_time_windows(batch):
    """
    batch: lista de (embeddings, label, meta)
    embeddings puede ser:
        - para caso simple: tensor (768,)
        - para secuencia: tensor (max_seq_len, 768) ya paddeado/submuestreado
          en __getitem__, por lo que todas las secuencias del batch ya
          tienen la misma longitud (self.max_seq_len).
    """
    is_sequence = len(batch[0][0].shape) == 2

    if not is_sequence:
        embeddings = torch.stack([item[0] for item in batch])  # (batch_size, 768)
        labels = torch.cat([item[1] for item in batch])        # (batch_size,)
        metas = [item[2] for item in batch]
        return embeddings, None, labels, metas
    else:
        # Todas las secuencias ya tienen longitud fija (max_seq_len) desde
        # __getitem__, por lo que no hace falta padding dinámico aquí.
        # Se reconstruye la máscara a partir de n_events real vs max_seq_len,
        # para que las posiciones de padding auténtico (ventanas con MENOS
        # eventos que max_seq_len) sigan enmascaradas correctamente.
        max_len   = batch[0][0].shape[0]
        embed_dim = batch[0][0].shape[1]

        padded_embeddings = torch.stack([item[0] for item in batch])  # (B, max_len, D)
        masks = torch.zeros(len(batch), max_len)
        labels = []

        for i, (emb, label, meta) in enumerate(batch):
            n_events = meta.get('n_events', max_len)
            real_len = min(n_events, max_len)
            masks[i, :real_len] = 1
            labels.append(label)

        labels = torch.cat(labels, dim=0)
        metas = [item[2] for item in batch]

        return padded_embeddings, masks, labels, metas