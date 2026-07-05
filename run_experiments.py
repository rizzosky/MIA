"""
run_experiments.py
------------------
Etapa 2 del pipeline — carga el dataset generado por prepare_dataset.py
y entrena/evalúa los tres modelos.

Uso:
    python run_experiments.py \
        --dataset    ./data/windows.pkl \
        --output_dir ./results \
        --device     mps \
        --models     transformer,bert,deeplog
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from models import build_model, ModelConfig
from dataset import TimeWindowDataset, collate_time_windows


def load_dataset(path: Path) -> dict:
    print(f"Cargando dataset desde {path}...")
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        print("  Formato antiguo detectado (lista plana).")
        print("  Aplicando split cronológico automático...")
        from prepare_dataset import chronological_split
        train, val, test = chronological_split(data)
        data = {
            "train":         train,
            "val":           val,
            "test":          test,
            "all":           data,
            "embedding_dim": len(data[0]["embedding"]),
        }
    else:
        print(f"  Creado: {data.get('created_at', 'N/A')}")
        print(f"  Ventana: {data.get('window_minutes', '?')} min  |  "
              f"Paso: {data.get('step_minutes', '?')} min")

    for split in ["train", "val", "test"]:
        ws    = data[split]
        n_pos = sum(1 for w in ws if w["has_anomaly"])
        print(f"  {split.capitalize():<6}: {len(ws):>5} ventanas  "
              f"(anómalas={n_pos:>4}, normales={len(ws)-n_pos:>5})")
    return data


def make_loaders(data, batch_size, max_seq_len, use_sequence):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = TimeWindowDataset(data[split], use_sequence=use_sequence,
                               max_seq_len=max_seq_len)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=0,
                                    collate_fn=collate_time_windows)
    print(f"\nDataLoaders — train: {len(loaders['train'])} | "
          f"val: {len(loaders['val'])} | test: {len(loaders['test'])} batches")
    return loaders["train"], loaders["val"], loaders["test"]


def compute_pos_weight(train_windows, device):
    n_pos  = sum(1 for w in train_windows if w["has_anomaly"])
    n_neg  = len(train_windows) - n_pos
    weight = n_neg / max(n_pos, 1)
    print(f"  pos_weight = {weight:.2f}  (neg={n_neg:,} / pos={n_pos:,})")
    return torch.tensor(weight).to(device)


def print_comparison_table(results):
    cols   = ["accuracy", "precision", "recall", "f1"]
    header = f"{'Modelo':<25} " + " ".join(f"{c:>10}" for c in cols)
    sep    = "─" * len(header)
    print(f"\n{'='*len(header)}")
    print("  TABLA COMPARATIVA DE RESULTADOS")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for name, m in results.items():
        print(f"{name:<25} " + " ".join(f"{m[c]:>10.3f}" for c in cols))
    print(f"{'='*len(header)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=Path, required=True)
    parser.add_argument("--output_dir",  type=Path, default=Path("./results"))
    parser.add_argument("--device",      type=str,  default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size",  type=int,  default=32)
    parser.add_argument("--max_seq_len", type=int,  default=20)
    parser.add_argument("--epochs",      type=int,  default=30)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--models",      type=str,
                        default="transformer,bert,deeplog")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data          = load_dataset(args.dataset)
    embedding_dim = data.get("embedding_dim") or len(data["train"][0]["embedding"])

    config = ModelConfig(
        embedding_dim=embedding_dim, hidden_dim=256, num_heads=4,
        num_layers=2, num_epochs=args.epochs, learning_rate=args.lr,
        use_sequence_embeddings=True, device=args.device,
        bert_model_name="bert-base-uncased",
    )

    train_loader, val_loader, test_loader = make_loaders(
        data, args.batch_size, args.max_seq_len, use_sequence=True)
    pos_weight = compute_pos_weight(data["train"], args.device)

    model_names = [m.strip() for m in args.models.split(",")]
    results     = {}

    for name in model_names:
        print(f"\n{'#'*60}\n  MODELO: {name.upper()}\n{'#'*60}")
        model    = build_model(name, config).to(args.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parámetros entrenables: {n_params:,}")

        best_epoch, best_f1, best_state = model.train_model(
            train_loader, val_loader, pos_weight)

        test_metrics = model.predict_model(test_loader)
        results[name] = {
            "accuracy":   round(float(test_metrics["accuracy"]),  4),
            "precision":  round(float(test_metrics["precision"]), 4),
            "recall":     round(float(test_metrics["recall"]),    4),
            "f1":         round(float(test_metrics["f1"]),        4),
            "best_epoch": best_epoch,
            "n_params":   n_params,
            "total_training_time_s": round(model.total_training_time, 2),
            "avg_epoch_time_s":       round(model.avg_epoch_time, 3),
            "n_epochs_run":           len(model.epoch_times),
            "inference_total_time_s": round(model.inference_time_total, 4),
            "inference_ms_per_window": round(model.inference_ms_per_sample, 4),
        }

        if best_state is not None:
            torch.save(best_state, args.output_dir / f"{name}_best.pt")

        cm = test_metrics.get("confusion_matrix")
        with open(args.output_dir / f"{name}_metrics.json", "w") as f:
            json.dump({**results[name],
                       "confusion_matrix": cm.tolist() if cm is not None else None,
                       "classification_report": test_metrics.get("classification_report")},
                      f, indent=2)

        del model
        if args.device == "mps":   torch.mps.empty_cache()
        elif args.device == "cuda": torch.cuda.empty_cache()

    print_comparison_table(results)
    with open(args.output_dir / "comparison_results.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "dataset": str(args.dataset),
                   "results": results}, f, indent=2)
    print(f"Resultados en {args.output_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
