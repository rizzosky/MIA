import pickle
with open("./modelos/task_scheduler_ASESP_2026-06-01_20-43-23/chunk_metadata.pkl", "rb") as f:
    windows = pickle.load(f)
lengths = [len(w["embeddings_sequence"]) for w in windows]
import numpy as np
print(np.percentile(lengths, [50, 75, 90, 95, 99]))