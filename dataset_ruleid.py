"""
dataset_ruleid.py
------------------
Dataset y collate_fn para el esquema de representación "rule_id only".

A diferencia de TimeWindowDataset (que maneja vectores de 768 dims
precalculados con BERT), este dataset maneja secuencias de índices
enteros (uno por evento, correspondiente a su rule_id en el
vocabulario). La capa de embedding que transforma estos índices en
vectores densos vive DENTRO del modelo (nn.Embedding) y se entrena
junto con el resto de la red, replicando el esquema clásico de
DeepLog/LogAnomaly/LogBERT sobre log keys categóricas.
"""

import torch
from torch.utils.data import Dataset


class RuleIdWindowDataset(Dataset):
    def __init__(self, windows: list, max_seq_len: int = 20):
        self.windows     = windows
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        seq    = window["rule_id_sequence"][: self.max_seq_len]
        label  = torch.tensor([window["has_anomaly"]], dtype=torch.float32)
        meta   = {
            "n_events":         window.get("n_events", 0),
            "mitre_techniques": window.get("mitre_techniques", []),
            "start_time":       window.get("start_time").isoformat()
                                if "start_time" in window else None,
            "end_time":         window.get("end_time").isoformat()
                                if "end_time" in window else None,
        }
        return torch.tensor(seq, dtype=torch.long), label, meta


def collate_ruleid_windows(batch):
    """
    batch: lista de (seq_tensor[long], label, meta)
    Devuelve (padded_seqs[long], mask[float], labels, metas)
    con el mismo formato (4 elementos) que collate_time_windows,
    para reusar el mismo train_model/validate_model sin cambios.
    PAD = índice 0 del vocabulario.
    """
    max_len = max(item[0].shape[0] for item in batch)
    padded  = torch.zeros(len(batch), max_len, dtype=torch.long)  # PAD=0
    mask    = torch.zeros(len(batch), max_len)
    labels  = []

    for i, (seq, label, _) in enumerate(batch):
        seq_len = seq.shape[0]
        if seq_len == 0:
            continue
        padded[i, :seq_len] = seq
        mask[i, :seq_len]   = 1
        labels.append(label)

    labels = torch.cat(labels, dim=0)
    metas  = [item[2] for item in batch]
    return padded, mask, labels, metas
