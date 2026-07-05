"""
run_logformer_ruleid.py
-------------------------
Entrenamiento en dos etapas de LogFormerAdapterLogKey (pre-train +
adapter tuning), análogo a run_logformer.py pero sobre la
representación categórica rule_id-only.

Uso:
    python run_logformer_ruleid.py \
        --dataset    ./data/windows_ruleid.pkl \
        --output_dir ./results_ruleid \
        --device     mps
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import torch

from models import ModelConfig
from models_ruleid import LogFormerAdapterLogKey
from run_experiments_ruleid import load_dataset, make_loaders, compute_pos_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=Path, required=True)
    parser.add_argument("--output_dir",     type=Path, default=Path("./results_ruleid"))
    parser.add_argument("--device",         type=str,  default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size",     type=int,  default=32)
    parser.add_argument("--max_seq_len",    type=int,  default=20)
    parser.add_argument("--embedding_dim",  type=int,  default=128)
    parser.add_argument("--hidden_dim",     type=int,  default=256)
    parser.add_argument("--pretrain_epochs", type=int, default=15)
    parser.add_argument("--tune_epochs",     type=int, default=15)
    parser.add_argument("--lr_pretrain",    type=float, default=1e-3)
    parser.add_argument("--lr_tune",        type=float, default=1e-3)
    parser.add_argument("--bottleneck_dim", type=int,  default=64)
    parser.add_argument("--num_layers",     type=int,  default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  LOGFORMER-LOGKEY — Pre-entrenamiento + Adapter Tuning")
    print(f"{'='*60}")

    data       = load_dataset(args.dataset)
    vocab_size = data["vocab_size"]

    config = ModelConfig(
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        num_heads=4, num_layers=args.num_layers,
        num_epochs=args.pretrain_epochs, learning_rate=args.lr_pretrain,
        device=args.device,
    )

    train_loader, val_loader, test_loader = make_loaders(
        data, args.batch_size, args.max_seq_len)
    pos_weight = compute_pos_weight(data["train"], args.device)

    model = LogFormerAdapterLogKey(
        vocab_size=vocab_size, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        bottleneck_dim=args.bottleneck_dim, config=config,
    ).to(args.device)

    # ── ETAPA 1 ──────────────────────────────────────────────────
    print(f"\n{'#'*60}\n  MODELO: LOGFORMER_LOGKEY_PRETRAIN\n{'#'*60}")
    params_1 = model.count_trainable_params()
    print(f"  Parámetros entrenables: {params_1['total']:,}")

    config.num_epochs    = args.pretrain_epochs
    config.learning_rate = args.lr_pretrain
    model.config = config

    best_epoch_1, best_f1_1, _ = model.train_model(
        train_loader, val_loader, pos_weight)
    metrics_1 = model.predict_model(test_loader)
    timing_1 = {
        "total_training_time_s": round(model.total_training_time, 2),
        "avg_epoch_time_s":       round(model.avg_epoch_time, 3),
        "n_epochs_run":           len(model.epoch_times),
        "inference_total_time_s": round(model.inference_time_total, 4),
        "inference_ms_per_window": round(model.inference_ms_per_sample, 4),
    }
    torch.save(model.state_dict(), args.output_dir / "logformer_logkey_pretrained.pt")

    # ── ETAPA 2 ──────────────────────────────────────────────────
    print(f"\n{'#'*60}\n  MODELO: LOGFORMER_LOGKEY_TUNED\n{'#'*60}")
    model.freeze_encoder_for_tuning()
    params_2 = model.count_trainable_params()
    print(f"  Parámetros entrenables: {params_2['total']:,} "
          f"(reducción {(1-params_2['total']/params_1['total'])*100:.1f}%)")

    config.num_epochs    = args.tune_epochs
    config.learning_rate = args.lr_tune
    model.config = config

    best_epoch_2, best_f1_2, _ = model.train_model(
        train_loader, val_loader, pos_weight)
    metrics_2 = model.predict_model(test_loader)
    timing_2 = {
        "total_training_time_s": round(model.total_training_time, 2),
        "avg_epoch_time_s":       round(model.avg_epoch_time, 3),
        "n_epochs_run":           len(model.epoch_times),
        "inference_total_time_s": round(model.inference_time_total, 4),
        "inference_ms_per_window": round(model.inference_ms_per_sample, 4),
    }
    torch.save(model.state_dict(), args.output_dir / "logformer_logkey_tuned.pt")

    # ── Guardar resultados ───────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset":   str(args.dataset),
        "vocab_size": vocab_size,
        "stage1_pretraining": {
            "best_epoch": best_epoch_1,
            "trainable_params": params_1["total"],
            "accuracy":  round(float(metrics_1["accuracy"]),  4),
            "precision": round(float(metrics_1["precision"]), 4),
            "recall":    round(float(metrics_1["recall"]),    4),
            "f1":        round(float(metrics_1["f1"]),        4),
            **timing_1,
        },
        "stage2_adapter_tuning": {
            "best_epoch": best_epoch_2,
            "trainable_params": params_2["total"],
            "accuracy":  round(float(metrics_2["accuracy"]),  4),
            "precision": round(float(metrics_2["precision"]), 4),
            "recall":    round(float(metrics_2["recall"]),    4),
            "f1":        round(float(metrics_2["f1"]),        4),
            **timing_2,
        },
    }

    with open(args.output_dir / "logformer_logkey_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\n  RESUMEN LOGFORMER-LOGKEY\n{'='*60}")
    print(f"  Etapa 1: F1 = {results['stage1_pretraining']['f1']}")
    print(f"  Etapa 2: F1 = {results['stage2_adapter_tuning']['f1']}")


if __name__ == "__main__":
    main()
