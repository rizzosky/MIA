import torch
from torch.utils.data import Dataset
import numpy as np

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
            # Si se usa secuencia, devolver la secuencia completa (con padding si es necesario)
            seq = np.array(window['embeddings_sequence'])
            seq = torch.FloatTensor(seq)
            if seq.shape[0] > self.max_seq_len:
                seq = seq[:self.max_seq_len]
            else:
                padding = torch.zeros(self.max_seq_len - seq.shape[0], seq.shape[1])
                seq = torch.cat([seq, padding], dim=0)
            embeddings = torch.FloatTensor(np.array(seq)) 
            label = torch.tensor([window['has_anomaly']], dtype=torch.float32)   
        else:
            # Siempre usar embedding promedio (evita problema de longitud variable)
            embeddings = torch.FloatTensor(window['embedding'])
            label = torch.tensor(window['has_anomaly'], dtype=torch.float32)

        # Devolver solo info serializable como tercer elemento
        meta = {
            'has_anomaly':     window['has_anomaly'],
            'mitre_techniques': window.get('mitre_techniques', []),
            'n_events':        window.get('n_events', 0),
        }

        return embeddings, label, meta

def collate_time_windows(batch):
    """
    batch: lista de (embeddings, label, meta)
    embeddings puede ser:
        - para caso simple: tensor (768,)
        - para secuencia: tensor (seq_len, 768)
    """
    
    # Verificamos si el primer elemento tiene 2 dimensiones (secuencia) o 1 (simple)
    is_sequence = len(batch[0][0].shape) == 2
    
    if len(batch[0][0].shape) == 1:
        # Caso embedding simple
        embeddings = torch.stack([item[0] for item in batch]) # (batch_size, 768)
        labels = torch.cat([item[1] for item in batch])     # (batch_size,)
        metas = [item[2] for item in batch]
        return embeddings, None, labels, metas
    else:
        # Caso secuencias de longitudes variables
        # Encontrar longitud máxima
        max_len = max(item[0].shape[0] for item in batch)
        embed_dim = batch[0][0].shape[1]
        # Crear batch con padding
        padded_embeddings = torch.zeros(len(batch), max_len, embed_dim)
        masks = torch.zeros(len(batch), max_len)
        labels = []

        for i, (emb, label, _) in enumerate(batch):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len] = emb
            masks[i, :seq_len] = 1
            labels.append(label)

        labels = torch.cat(labels, dim=0)
        metas = [item[2] for item in batch]

        return padded_embeddings, masks, labels, metas