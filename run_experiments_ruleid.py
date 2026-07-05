"""
run_experiments_ruleid.py
---------------------------
Entrena y evalúa TimeAwareTransformerLogKey y DeepLogBaselineLogKey
sobre el dataset de representación "rule_id only", generado por
prepare_dataset_ruleid.py.

Uso:
    python run_experiments_ruleid.py \
        --dataset    ./data/windows_ruleid.pkl \
        --output_dir ./results_ruleid \
        --device     mps \
        --models     transformer_logkey,deeplog_logkey
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from models import ModelConfig
from models_ruleid import build_model_ruleid
from dataset_ruleid import RuleIdWindowDataset, collate_ruleid_windows


def load_dataset(path: Path) -> dict:
    print(f"Cargando dataset desde {path}...")
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"  Representación: {data.get('representation', '?')}")
    print(f"  Vocabulario: {data.get('vocab_size', '?')} tokens")
    for split in ["train", "val", "test"]:
        ws    = data[split]
        n_pos = sum(1 for w in ws if w["has_anomaly"])
        print(f"  {split.capitalize():<6}: {len(ws):>5} ventanas  "
              f"(anómalas={n_pos:>4}, normales={len(ws)-n_pos:>5})")
    return data


def make_loaders(data, batch_size, max_seq_len):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = RuleIdWindowDataset(data[split], max_seq_len=max_seq_len)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=0, collate_fn=collate_ruleid_windows,
        )
    print(f"\nDataLoaders — train: {len(loaders['train'])} | "
          f"val: {len(loaders['val'])} | test: {len(loaders['test'])} batches")
    return loaders["train"], loaders["val"], loaders["test"]


def compute_pos_weight(train_windows, device):
    n_pos  = sum(1 for w in train_windows if w["has_anomaly"])
    n_neg  = len(train_windows) - n_pos
    weight = n_neg / max(n_pos, 1)
    print(f"  pos_weight = {weight:.2f}  (neg={n_neg:,} / pos={n_pos:,})")
    return torch.tensor(weight).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=Path, required=True)
    parser.add_argument("--output_dir",    type=Path, default=Path("./results_ruleid"))
    parser.add_argument("--device",        type=str,  default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size",    type=int,  default=32)
    parser.add_argument("--max_seq_len",   type=int,  default=20)
    parser.add_argument("--epochs",        type=int,  default=30)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int,  default=128)
    parser.add_argument("--hidden_dim",    type=int,  default=256)
    parser.add_argument("--models",        type=str,
                        default="transformer_logkey,deeplog_logkey")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  EXPERIMENTO: representación rule_id-only")
    print(f"{'='*60}")

    data       = load_dataset(args.dataset)
    vocab_size = data["vocab_size"]

    config = ModelConfig(
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        num_heads=4, num_layers=2, num_epochs=args.epochs,
        learning_rate=args.lr, device=args.device,
    )

    train_loader, val_loader, test_loader = make_loaders(
        data, args.batch_size, args.max_seq_len)
    pos_weight = compute_pos_weight(data["train"], args.device)

    model_names = [m.strip() for m in args.models.split(",")]
    results     = {}

    for name in model_names:
        print(f"\n{'#'*60}\n  MODELO: {name.upper()}\n{'#'*60}")
        model = build_model_ruleid(
            name, vocab_size, config,
            embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim
        ).to(args.device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parámetros entrenables: {n_params:,} "
              f"(vocab_size={vocab_size}, embedding_dim={args.embedding_dim})")

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
            "vocab_size": vocab_size,
            "total_training_time_s": round(model.total_training_time, 2),
            "avg_epoch_time_s":       round(model.avg_epoch_time, 3),
            "n_epochs_run":           len(model.epoch_times),
            "inference_total_time_s": round(model.inference_time_total, 4),
            "inference_ms_per_window": round(model.inference_ms_per_sample, 4),
        }

        if best_state is not None:
            torch.save(best_state, args.output_dir / f"{name}_best.pt")

        del model
        if args.device == "mps":   torch.mps.empty_cache()
        elif args.device == "cuda": torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  RESULTADOS — representación rule_id-only\n{'='*60}")
    for name, m in results.items():
        print(f"  {name:<22} F1={m['f1']:.3f}  Prec={m['precision']:.3f}  "
              f"Rec={m['recall']:.3f}  params={m['n_params']:,}")

    with open(args.output_dir / "comparison_results_ruleid.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset":   str(args.dataset),
            "vocab_size": vocab_size,
            "results":   results,
        }, f, indent=2)
    print(f"\nGuardado en {args.output_dir / 'comparison_results_ruleid.json'}")


if __name__ == "__main__":
    main()
