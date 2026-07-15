"""
regenerar_tablas.py
-------------------
Regenera grid_tables.tex a partir de los CSV y el JSON que el grid
search ya guardó, SIN reentrenar nada.

Uso (desde la carpeta del proyecto, junto a grid_search.py corregido):
    python regenerar_tablas.py --grid_dir results/grid
"""

import csv
import json
import argparse
from pathlib import Path

from grid_search import write_latex


def cargar_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k == "learning_rate":
                    row[k] = float(v)
                elif k == "val_f1":
                    row[k] = float(v)
                elif k in ("batch_size", "hidden_dim", "best_epoch",
                           "n_epochs_run", "n_params"):
                    row[k] = int(float(v))
                elif k == "train_time_s":
                    row[k] = float(v)
                else:
                    row[k] = v
            rows.append(row)
    rows.sort(key=lambda r: r["val_f1"], reverse=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_dir", type=Path,
                   default=Path("./results/grid"))
    args = p.parse_args()

    with open(args.grid_dir / "grid_summary.json") as f:
        summary = json.load(f)["summary"]

    all_results = {}
    for csv_path in sorted(args.grid_dir.glob("grid_*.csv")):
        name = csv_path.stem.replace("grid_", "")
        all_results[name] = cargar_csv(csv_path)
        print(f"  [OK] {csv_path.name}: "
              f"{len(all_results[name])} configuraciones")

    out = args.grid_dir / "grid_tables.tex"
    write_latex(all_results, summary, out)
    print(f"  [OK] {out} regenerado.")


if __name__ == "__main__":
    main()
