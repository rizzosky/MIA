"""
grid_search.py
--------------
Búsqueda por grilla de hiperparámetros sobre el conjunto de VALIDACIÓN.

Metodología:
    1. Para cada configuración de la grilla se entrena el modelo con
       train, se selecciona por F1 de validación (el mismo criterio de
       early stopping ya usado en train_model).
    2. El conjunto de TEST se evalúa UNA sola vez, únicamente para la
       configuración ganadora de cada modelo. Esto evita el sesgo de
       selección sobre test.
    3. La semilla se fija en cada corrida para que las diferencias
       entre configuraciones no se deban a la inicialización.

Uso (dataset de Windows, ~1 hora total en M5 Pro / MPS):
    python grid_search.py --dataset data/windows.pkl --device mps \
        --models transformer,deeplog,bert --output_dir results/grid

Salidas en output_dir:
    grid_<modelo>.csv          — todas las corridas, ordenadas por val F1
    grid_summary.json          — config ganadora + métricas de test
    grid_tables.tex            — tablas LaTeX listas para el documento
"""

import csv
import json
import time
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import build_model, ModelConfig
from dataset import TimeWindowDataset, collate_time_windows
from run_experiments import load_dataset, compute_pos_weight

# ─────────────────────────────────────────────────────────────────
# Grillas por modelo.
# - hidden_dim no aplica a bert (su hidden size es fijo: 768).
# - dropout y paciencia de early stopping quedan fijos (0,1 y 7),
#   valores por convención (Devlin et al., 2019; Vaswani et al., 2017).
# ─────────────────────────────────────────────────────────────────
GRIDS = {
    "transformer": {
        "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4],
        "batch_size":    [16, 32, 64],
        "hidden_dim":    [128, 256, 512],
    },
    "deeplog": {
        "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4],
        "batch_size":    [16, 32, 64],
        "hidden_dim":    [128, 256, 512],
    },
    "bert": {
        "learning_rate": [1e-5, 2e-5, 3e-5],
        "batch_size":    [16, 32],
    },
}

# Grilla reducida (--quick): solo learning rate y batch size
QUICK_GRIDS = {
    "transformer": {"learning_rate": [1e-5, 2e-5, 1e-4],
                    "batch_size": [16, 32]},
    "deeplog":     {"learning_rate": [1e-5, 2e-5, 1e-4],
                    "batch_size": [16, 32]},
    "bert":        {"learning_rate": [1e-5, 2e-5, 3e-5],
                    "batch_size": [32]},
}

# Grilla arquitectónica (--arch): varía num_heads y num_layers
# manteniendo fija la configuración común (ceteris paribus).
# hidden_dim=256 es divisible por todas las cabezas probadas.
# No aplica a bert (arquitectura preentrenada fija: 12 capas,
# 12 cabezas). Usar un --output_dir distinto para no pisar los
# CSV de la grilla principal.
ARCH_GRIDS = {
    "transformer": {"learning_rate": [2e-5], "batch_size": [32],
                    "hidden_dim": [256],
                    "num_heads": [2, 4, 8],
                    "num_layers": [1, 2, 4]},
    "deeplog":     {"learning_rate": [2e-5], "batch_size": [32],
                    "hidden_dim": [256],
                    "num_layers": [1, 2, 3]},
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data, batch_size, max_seq_len):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = TimeWindowDataset(data[split], use_sequence=True,
                               max_seq_len=max_seq_len)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=0,
                                    collate_fn=collate_time_windows)
    return loaders["train"], loaders["val"], loaders["test"]


def free_memory(model, device):
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def retrain_and_test(name, params, data, args, embedding_dim, pos_weight):
    """
    Reentrena el modelo con la configuración dada — replicando
    exactamente el procedimiento de run_experiments.py — y evalúa
    sobre test UNA única vez con el modelo resultante.

    Nota: no se reutiliza el best_state devuelto por train_model
    porque puede ser None (la captura del estado en TrainingMixin
    ocurre después de actualizar el early stopping, por lo que rara
    vez se dispara); evaluar el modelo entrenado directamente es el
    mismo comportamiento que run_experiments.py.
    """
    set_seed(args.seed)
    config = ModelConfig(
        embedding_dim=embedding_dim,
        hidden_dim=params.get("hidden_dim", 256),
        num_heads=params.get("num_heads", 4),
        num_layers=params.get("num_layers", 2),
        num_epochs=args.epochs,
        learning_rate=params["learning_rate"],
        use_sequence_embeddings=True, device=args.device,
        bert_model_name="bert-base-uncased",
    )
    train_loader, val_loader, test_loader = make_loaders(
        data, params["batch_size"], args.max_seq_len)
    model = build_model(name, config).to(args.device)
    t0 = time.time()
    model.train_model(train_loader, val_loader, pos_weight)
    test_metrics = model.predict_model(test_loader)
    test_row = {k: round(float(test_metrics[k]), 4)
                for k in ["accuracy", "precision", "recall", "f1"]}
    test_row["final_train_time_s"] = round(
        getattr(model, "total_training_time", time.time() - t0), 1)
    test_row["inference_ms_per_window"] = round(
        getattr(model, "inference_ms_per_sample", 0.0), 3)
    print(f"  [tiempos] entrenamiento final: "
          f"{test_row['final_train_time_s']}s | inferencia: "
          f"{test_row['inference_ms_per_window']} ms/ventana")
    free_memory(model, args.device)
    return test_row


def cargar_csv(path: Path) -> list:
    """Carga un grid_<modelo>.csv con los tipos correctos."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in ("learning_rate", "val_f1", "train_time_s"):
                    row[k] = float(v)
                elif k in ("batch_size", "hidden_dim", "best_epoch",
                           "n_epochs_run", "n_params"):
                    row[k] = int(float(v))
                else:
                    row[k] = v
            rows.append(row)
    rows.sort(key=lambda r: r["val_f1"], reverse=True)
    return rows


def run_grid_for_model(name, grid, data, args, embedding_dim, pos_weight):
    keys    = list(grid.keys())
    combos  = list(itertools.product(*(grid[k] for k in keys)))
    results = []
    best    = {"val_f1": -1.0, "config": None}

    print(f"\n{'#'*64}\n  GRID SEARCH: {name.upper()} "
          f"({len(combos)} configuraciones)\n{'#'*64}")
    t_model = time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{name} {i}/{len(combos)}] {params}")
        set_seed(args.seed)

        config = ModelConfig(
            embedding_dim=embedding_dim,
            hidden_dim=params.get("hidden_dim", 256),
            num_heads=params.get("num_heads", 4),
            num_layers=params.get("num_layers", 2),
            num_epochs=args.epochs,
            learning_rate=params["learning_rate"],
            use_sequence_embeddings=True,
            device=args.device,
            bert_model_name="bert-base-uncased",
        )
        train_loader, val_loader, _ = make_loaders(
            data, params["batch_size"], args.max_seq_len)

        model = build_model(name, config).to(args.device)
        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

        t0 = time.time()
        best_epoch, best_val_f1, best_state = model.train_model(
            train_loader, val_loader, pos_weight)
        elapsed = time.time() - t0

        row = {**params, "val_f1": round(float(best_val_f1 or 0), 4),
               "best_epoch": best_epoch,
               "n_epochs_run": len(model.epoch_times),
               "n_params": n_params,
               "train_time_s": round(elapsed, 1)}
        results.append(row)
        avg = (time.time() - t_model) / i
        eta_min = avg * (len(combos) - i) / 60
        print(f"  → val F1 = {row['val_f1']:.4f} "
              f"(época {best_epoch}, {elapsed:.0f}s) | "
              f"transcurrido: {(time.time()-t_model)/60:.1f} min | "
              f"ETA {name}: {eta_min:.1f} min")

        if row["val_f1"] > best["val_f1"]:
            best = {"val_f1": row["val_f1"], "config": params}
        free_memory(model, args.device)

    # ── Evaluación única sobre test con la config ganadora ──────
    # Se reentrena la config ganadora y se evalúa el modelo
    # resultante, replicando el procedimiento de run_experiments.py.
    print(f"\n[{name}] Grilla completada en "
          f"{(time.time()-t_model)/60:.1f} min "
          f"({len(combos)} configuraciones)")
    print(f"[{name}] Config ganadora por val F1: {best['config']} "
          f"(val F1 = {best['val_f1']:.4f})")
    print(f"[{name}] Reentrenando y evaluando sobre TEST (única vez)...")
    test_row = retrain_and_test(name, best["config"], data, args,
                                embedding_dim, pos_weight)

    results.sort(key=lambda r: r["val_f1"], reverse=True)
    return results, best["config"], best["val_f1"], test_row


def write_latex(all_results, summary, out_path, top_n=5, suffix=""):
    """Genera tablas LaTeX: top-N configs por modelo + resumen final."""
    lines = []
    for name, rows in all_results.items():
        keys = [k for k in rows[0] if k not in
                ("val_f1", "best_epoch", "n_epochs_run",
                 "n_params", "train_time_s")]
        header = " & ".join(k.replace("_", r"\_") for k in keys)
        lines += [
            f"% --- Top {top_n} configuraciones: {name} ---",
            r"\begin{table}[h]", r"\centering",
            f"\\caption{{Búsqueda de hiperparámetros — {name} "
            f"(top {top_n} por F1 de validación)}}",
            f"\\label{{tab:grid_{name}{suffix}}}",
            r"\begin{tabular}{%s}" % ("l" * len(keys) + "rr"),
            r"\toprule",
            header + r" & \textbf{Val F1} & \textbf{Épocas} \\",
            r"\midrule",
        ]
        for r_ in rows[:top_n]:
            vals = " & ".join(
                (f"{r_[k]:.0e}" if k == "learning_rate" else str(r_[k]))
                for k in keys)
            lines.append(f"{vals} & {r_['val_f1']:.4f} & "
                         f"{r_['n_epochs_run']} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"% ─── Resumen: configuración ganadora y test ───",
        r"\begin{table}[h]", r"\centering",
        r"\caption{Configuraciones seleccionadas por validación y su "
        r"desempeño en test (evaluado una única vez)}",
        f"\\label{{tab:grid_resumen{suffix}}}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"\textbf{Modelo} & \textbf{Config. ganadora} & "
        r"\textbf{Val F1} & \textbf{Prec.} & \textbf{Rec.} & "
        r"\textbf{F1 (test)} \\",
        r"\midrule",
    ]
    for name, s in summary.items():
        cfg = ", ".join(
            (f"{k}={v:.0e}" if k == "learning_rate" else f"{k}={v}")
            .replace("_", r"\_")
            for k, v in s["best_config"].items())
        t = s["test_metrics"]
        lines.append(f"{name} & {cfg} & {s['best_val_f1']:.4f} & "
                     f"{t['precision']:.3f} & {t['recall']:.3f} & "
                     f"{t['f1']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=Path, required=True)
    parser.add_argument("--output_dir",  type=Path,
                        default=Path("./results/grid"))
    parser.add_argument("--device",      type=str, default="mps",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--models",      type=str,
                        default="transformer,deeplog,bert")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Grilla reducida (solo LR y batch size)")
    parser.add_argument("--arch", action="store_true",
                        help="Grilla arquitectónica: num_heads y "
                             "num_layers con el resto fijo. Usar un "
                             "--output_dir distinto al de la grilla "
                             "principal.")
    parser.add_argument("--from_summary", type=Path, default=None,
                        help="Ruta a un grid_summary.json existente: "
                             "NO repite la grilla, solo reentrena las "
                             "configs ganadoras y reevalúa test.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grids = (ARCH_GRIDS if args.arch
             else QUICK_GRIDS if args.quick else GRIDS)
    label_suffix = "_arch" if args.arch else ""

    data = load_dataset(args.dataset)
    embedding_dim = (data.get("embedding_dim")
                     or len(data["train"][0]["embedding"]))
    pos_weight = compute_pos_weight(data["train"], args.device)

    all_results, summary = {}, {}
    t_global = time.time()

    # ── Modo reevaluación: solo el test final de las ganadoras ──
    if args.from_summary is not None:
        with open(args.from_summary) as f:
            summary = json.load(f)["summary"]
        for name, s in summary.items():
            print(f"\n{'#'*64}\n  REEVALUACIÓN TEST: {name.upper()} — "
                  f"config {s['best_config']}\n{'#'*64}")
            s["test_metrics"] = retrain_and_test(
                name, s["best_config"], data, args,
                embedding_dim, pos_weight)
        for csv_path in sorted(args.output_dir.glob("grid_*.csv")):
            all_results[csv_path.stem.replace("grid_", "")] = \
                cargar_csv(csv_path)
        with open(args.output_dir / "grid_summary.json", "w") as f:
            json.dump({"seed": args.seed,
                       "max_seq_len": args.max_seq_len,
                       "dataset": str(args.dataset),
                       "elapsed_total_s": round(time.time() - t_global, 1),
                       "summary": summary}, f, indent=2)
        if all_results:
            write_latex(all_results, summary,
                        args.output_dir / "grid_tables.tex",
                        suffix=label_suffix)
        print(f"\n{'='*64}\n  RESUMEN (test reevaluado)\n{'='*64}")
        for name, s in summary.items():
            print(f"  {name:<12} val F1={s['best_val_f1']:.4f}  "
                  f"test F1={s['test_metrics']['f1']:.4f}  "
                  f"config={s['best_config']}")
        print(f"  Tiempo total: {(time.time()-t_global)/60:.1f} min")
        return

    for name in [m.strip() for m in args.models.split(",")]:
        if name not in grids:
            print(f"[SKIP] '{name}' no tiene grilla definida.")
            continue
        rows, best_cfg, best_val_f1, test_row = run_grid_for_model(
            name, grids[name], data, args, embedding_dim, pos_weight)
        all_results[name] = rows
        summary[name] = {"best_config": best_cfg,
                         "best_val_f1": best_val_f1,
                         "test_metrics": test_row}

        with open(args.output_dir / f"grid_{name}.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  [OK] grid_{name}.csv")

    with open(args.output_dir / "grid_summary.json", "w") as f:
        json.dump({"seed": args.seed, "max_seq_len": args.max_seq_len,
                   "dataset": str(args.dataset),
                   "elapsed_total_s": round(time.time() - t_global, 1),
                   "summary": summary}, f, indent=2)

    write_latex(all_results, summary,
                args.output_dir / "grid_tables.tex",
                suffix=label_suffix)

    print(f"\n{'='*64}")
    print("  RESUMEN GRID SEARCH")
    print(f"{'='*64}")
    for name, s in summary.items():
        print(f"  {name:<12} val F1={s['best_val_f1']:.4f}  "
              f"test F1={s['test_metrics']['f1']:.4f}  "
              f"config={s['best_config']}")
    print(f"  Tiempo total: {(time.time()-t_global)/60:.1f} min")
    print(f"  Salidas en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()