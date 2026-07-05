"""
Verifica la distribución de eventos por ventana en un dataset ya
generado, para diagnosticar si max_seq_len está truncando demasiado.
"""
import pickle
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/windows_pfsense_500k.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

all_windows = data["all"]
n_events = np.array([w["n_events"] for w in all_windows])

print(f"Dataset: {path}")
print(f"Total ventanas: {len(all_windows):,}")
print(f"\nDistribución de eventos por ventana:")
print(f"  Media:     {n_events.mean():.1f}")
print(f"  Mediana:   {np.median(n_events):.1f}")
print(f"  Std:       {n_events.std():.1f}")
print(f"  Min:       {n_events.min()}")
print(f"  Max:       {n_events.max()}")
for p in [50, 75, 90, 95, 99]:
    print(f"  P{p}:       {np.percentile(n_events, p):.1f}")

max_seq_len = 20
pct_truncated = (n_events > max_seq_len).mean() * 100
print(f"\nCon max_seq_len={max_seq_len}:")
print(f"  Ventanas truncadas: {pct_truncated:.1f}%")
print(f"  Eventos perdidos en promedio (ventanas truncadas): "
      f"{(n_events[n_events > max_seq_len] - max_seq_len).mean():.1f}")
