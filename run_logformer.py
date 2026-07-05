"""
run_logformer.py
-----------------
Entrenamiento de LogFormerAdapter en dos etapas, replicando el
pipeline propuesto en el paper original (Guo et al., 2024):

    Etapa 1 — Pre-entrenamiento:
        Se entrena el Log-Attention encoder completo (capas base +
        adapters) sobre el dominio fuente, definido en este trabajo
        como el conjunto de entrenamiento completo (normal + incidente).

    Etapa 2 — Adapter-based tuning:
        Se congelan las capas base del encoder pre-entrenado y se
        reentrenan únicamente los adapters y el clasificador final,
        replicando la transferencia de conocimiento con bajo costo
        de parámetros del paper original.

Uso:
    python run_logformer.py \
        --dataset    ./data/windows.pkl \
        --output_dir ./results \
        --device     mps
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from models import LogFormerAdapter, ModelConfig
from dataset import TimeWindowDataset, collate_time_windows
from run_experiments import load_dataset, make_loaders, compute_pos_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=Path, required=True)
    parser.add_argument("--output_dir",   type=Path, default=Path("./results"))
    parser.add_argument("--device",       type=str,  default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size",   type=int,  default=32)
    parser.add_argument("--max_seq_len",  type=int,  default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=15)
    parser.add_argument("--tune_epochs",     type=int, default=15)
    parser.add_argument("--lr_pretrain",  type=float, default=2e-5)
    parser.add_argument("--lr_tune",      type=float, default=1e-4)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--num_layers",   type=int,  default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  LOGFORMER — Pre-entrenamiento + Adapter Tuning")
    print(f"{'='*60}")

    # ── Cargar datos ──────────────────────────────────────────────
    data          = load_dataset(args.dataset)
    embedding_dim = data.get("embedding_dim") or len(data["train"][0]["embedding"])

    config = ModelConfig(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=args.num_layers,
        num_epochs=args.pretrain_epochs,  # se sobreescribe por etapa
        learning_rate=args.lr_pretrain,
        use_sequence_embeddings=True,
        device=args.device,
    )

    train_loader, val_loader, test_loader = make_loaders(
        data, args.batch_size, args.max_seq_len, use_sequence=True)
    pos_weight = compute_pos_weight(data["train"], args.device)

    model = LogFormerAdapter(
        config, num_layers=args.num_layers,
        bottleneck_dim=args.bottleneck_dim
    ).to(args.device)

    # ════════════════════════════════════════════════════════════
    # ETAPA 1 — PRE-ENTRENAMIENTO (encoder completo entrenable)
    # ════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  MODELO: LOGFORMER_PRETRAIN")
    print(f"{'#'*60}")

    params_stage1 = model.count_trainable_params()
    print(f"  Parámetros entrenables: {params_stage1['total']:,}")
    print(f"    - Capas base : {params_stage1['base_layers']:,}")
    print(f"    - Adapters   : {params_stage1['adapters']:,}")
    print(f"    - Clasificador: {params_stage1['classifier']:,}")

    config.num_epochs    = args.pretrain_epochs
    config.learning_rate = args.lr_pretrain
    model.config = config

    best_epoch_1, best_f1_1, best_state_1 = model.train_model(
        train_loader, val_loader, pos_weight
    )

    print(f"\n── Evaluación tras pre-entrenamiento ──")
    metrics_stage1 = model.predict_model(test_loader)
    timing_stage1 = {
        "total_training_time_s": round(model.total_training_time, 2),
        "avg_epoch_time_s":       round(model.avg_epoch_time, 3),
        "n_epochs_run":           len(model.epoch_times),
        "inference_total_time_s": round(model.inference_time_total, 4),
        "inference_ms_per_window": round(model.inference_ms_per_sample, 4),
    }

    torch.save(model.state_dict(), args.output_dir / "logformer_pretrained.pt")

    # ════════════════════════════════════════════════════════════
    # ETAPA 2 — ADAPTER-BASED TUNING (encoder congelado)
    # ════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  MODELO: LOGFORMER_TUNED")
    print(f"{'#'*60}")

    model.freeze_encoder_for_tuning()

    params_stage2 = model.count_trainable_params()
    print(f"  Parámetros entrenables: {params_stage2['total']:,}")
    print(f"    - Capas base (congeladas): {params_stage1['base_layers']:,} -> 0")
    print(f"    - Adapters (entrenables) : {params_stage2['adapters']:,}")
    print(f"    - Clasificador           : {params_stage2['classifier']:,}")
    print(f"  Reducción de parámetros entrenables: "
          f"{(1 - params_stage2['total']/params_stage1['total'])*100:.1f}%")

    config.num_epochs    = args.tune_epochs
    config.learning_rate = args.lr_tune
    model.config = config

    best_epoch_2, best_f1_2, best_state_2 = model.train_model(
        train_loader, val_loader, pos_weight
    )

    print(f"\n── Evaluación final tras adapter tuning ──")
    metrics_stage2 = model.predict_model(test_loader)
    timing_stage2 = {
        "total_training_time_s": round(model.total_training_time, 2),
        "avg_epoch_time_s":       round(model.avg_epoch_time, 3),
        "n_epochs_run":           len(model.epoch_times),
        "inference_total_time_s": round(model.inference_time_total, 4),
        "inference_ms_per_window": round(model.inference_ms_per_sample, 4),
    }

    torch.save(model.state_dict(), args.output_dir / "logformer_tuned.pt")

    # ── Guardar resultados ───────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset":   str(args.dataset),
        "stage1_pretraining": {
            "best_epoch": best_epoch_1,
            "trainable_params": params_stage1["total"],
            "accuracy":  round(float(metrics_stage1["accuracy"]),  4),
            "precision": round(float(metrics_stage1["precision"]), 4),
            "recall":    round(float(metrics_stage1["recall"]),    4),
            "f1":        round(float(metrics_stage1["f1"]),        4),
            **timing_stage1,
        },
        "stage2_adapter_tuning": {
            "best_epoch": best_epoch_2,
            "trainable_params": params_stage2["total"],
            "accuracy":  round(float(metrics_stage2["accuracy"]),  4),
            "precision": round(float(metrics_stage2["precision"]), 4),
            "recall":    round(float(metrics_stage2["recall"]),    4),
            "f1":        round(float(metrics_stage2["f1"]),        4),
            **timing_stage2,
        },
    }

    with open(args.output_dir / "logformer_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("  RESUMEN LOGFORMER")
    print(f"{'='*60}")
    print(f"  Etapa 1 (pre-train, {params_stage1['total']:,} params): "
          f"F1 = {results['stage1_pretraining']['f1']}")
    print(f"  Etapa 2 (adapter tuning, {params_stage2['total']:,} params): "
          f"F1 = {results['stage2_adapter_tuning']['f1']}")
    print(f"  Resultados guardados en {args.output_dir / 'logformer_metrics.json'}")


if __name__ == "__main__":
    main()
