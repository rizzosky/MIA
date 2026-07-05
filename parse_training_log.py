"""
parse_training_log.py
---------------------
Extrae las métricas de entrenamiento del log de consola y las guarda
como CSV y figuras PNG para incluir en la tesis.

Uso:
    # Primero guardar el log en un archivo:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python run_experiments.py ... > results/training.log 2>&1

    # Luego parsear:
    python parse_training_log.py --log results/training.log --output_dir results/curves
"""

import re
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# Parser del log
# ─────────────────────────────────────────────────────────────────────────────

MODEL_HEADER = re.compile(r"MODELO:\s+(\w+)", re.IGNORECASE)
EPOCH_LINE   = re.compile(
    r"Época\s+(\d+)\s*[—-]\s*Loss:\s*([\d.]+)\s*\|"
    r"\s*Val Loss:\s*([\d.]+)\s*\|"
    r"\s*Val Acc:\s*([\d.]+)\s*\|"
    r"\s*Val Prec:\s*([\d.]+)\s*\|"
    r"\s*Val Rec:\s*([\d.]+)\s*\|"
    r"\s*Val F1:\s*([\d.]+)"
)
TEST_LINE = re.compile(
    r"Test Acc:\s*([\d.]+)\s*\|"
    r"\s*Prec:\s*([\d.]+)\s*\|"
    r"\s*Rec:\s*([\d.]+)\s*\|"
    r"\s*F1:\s*([\d.]+)"
)


def parse_log(log_path: Path) -> dict:
    """
    Parsea el log de entrenamiento y devuelve un dict:
        {
            "transformer": {
                "epochs": [{"epoch": 1, "loss": ..., "val_acc": ..., ...}, ...],
                "test":   {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
            },
            ...
        }
    """
    results     = {}
    current     = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Detectar inicio de modelo
            m = MODEL_HEADER.search(line)
            if m:
                current = m.group(1).lower()
                results[current] = {"epochs": [], "test": {}}
                continue

            if current is None:
                continue

            # Línea de época
            m = EPOCH_LINE.search(line)
            if m:
                results[current]["epochs"].append({
                    "epoch":    int(m.group(1)),
                    "loss":     float(m.group(2)),
                    "val_loss": float(m.group(3)),
                    "val_acc":  float(m.group(4)),
                    "val_prec": float(m.group(5)),
                    "val_rec":  float(m.group(6)),
                    "val_f1":   float(m.group(7)),
                })
                continue

            # Línea de test
            m = TEST_LINE.search(line)
            if m:
                results[current]["test"] = {
                    "accuracy":  float(m.group(1)),
                    "precision": float(m.group(2)),
                    "recall":    float(m.group(3)),
                    "f1":        float(m.group(4)),
                }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Exportar CSVs
# ─────────────────────────────────────────────────────────────────────────────

def export_csvs(results: dict, output_dir: Path):
    """Guarda un CSV por modelo con las métricas por época."""
    fields = ["epoch", "loss", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1"]

    for model_name, data in results.items():
        if not data["epochs"]:
            continue
        path = output_dir / f"{model_name}_training_curves.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data["epochs"])
        print(f"  [OK] {path.name}")

    # CSV resumen de test
    test_path = output_dir / "test_results_summary.csv"
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy",
                                                "precision", "recall", "f1"])
        writer.writeheader()
        for model_name, data in results.items():
            if data["test"]:
                writer.writerow({"model": model_name, **data["test"]})
    print(f"  [OK] {test_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "transformer":        "#4C9BE8",
    "bert":               "#E8614C",
    "deeplog":            "#4CAF50",
    "logformer_pretrain": "#9B6FD8",
    "logformer_tuned":    "#D89B2E",
}
LABELS = {
    "transformer":        "TimeAwareTransformer",
    "bert":               "TimeAwareBERT",
    "deeplog":            "DeepLog Baseline",
    "logformer_pretrain": "LogFormer (pre-train)",
    "logformer_tuned":    "LogFormer (adapter tuning)",
}


def plot_metric_comparison(results: dict, metric: str,
                            ylabel: str, title: str,
                            output_path: Path):
    """Curvas de una métrica para todos los modelos en un solo gráfico."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name, data in results.items():
        if not data["epochs"]:
            continue
        epochs = [e["epoch"] for e in data["epochs"] if metric in e]
        values = [e[metric]  for e in data["epochs"] if metric in e]
        if not values:
            print(f"  [SKIP] {model_name}: sin datos para métrica '{metric}'")
            continue
        ax.plot(epochs, values,
                label=LABELS.get(model_name, model_name),
                color=COLORS.get(model_name, "gray"),
                linewidth=1.8, marker="o", markersize=3)

    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path.name}")


def plot_per_model(results: dict, output_dir: Path):
    """Un gráfico por modelo con loss + val_f1 en ejes duales."""
    for model_name, data in results.items():
        if not data["epochs"]:
            continue

        epochs_full = [e for e in data["epochs"] if "val_f1" in e and "loss" in e]
        if not epochs_full:
            print(f"  [SKIP] {model_name}: épocas incompletas para detalle")
            continue

        epochs     = [e["epoch"]    for e in epochs_full]
        losses     = [e["loss"]     for e in epochs_full]
        val_losses = [e.get("val_loss") for e in epochs_full]
        val_f1s    = [e["val_f1"]   for e in epochs_full]
        val_rec    = [e["val_rec"]  for e in epochs_full]
        val_pre    = [e["val_prec"] for e in epochs_full]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = COLORS.get(model_name, "gray")
        label = LABELS.get(model_name, model_name)

        ax1.set_xlabel("Época")
        ax1.set_ylabel("Loss", color="gray")
        ax1.plot(epochs, losses, color="gray", linewidth=1.5,
                 linestyle="--", label="Train Loss")
        if any(v is not None for v in val_losses):
            ax1.plot(epochs, val_losses, color="black", linewidth=1.5,
                     linestyle=":", label="Val Loss")
        ax1.tick_params(axis="y", labelcolor="gray")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Métricas de validación")
        ax2.plot(epochs, val_f1s, color=color,       linewidth=2,
                 marker="o", markersize=4, label="Val F1")
        ax2.plot(epochs, val_rec, color=color,       linewidth=1.2,
                 linestyle=":", label="Val Recall")
        ax2.plot(epochs, val_pre, color=color,       linewidth=1.2,
                 linestyle="-.", label="Val Precision")
        ax2.set_ylim(0.5, 1.02)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        ax1.set_title(f"Curvas de entrenamiento — {label}")
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        path = output_dir / f"{model_name}_training_detail.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {path.name}")


def plot_test_comparison(results: dict, output_dir: Path):
    """Gráfico de barras con las métricas de test de todos los modelos."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    models  = [m for m in results if results[m]["test"]]
    labels  = [LABELS.get(m, m) for m in models]
    n_models = len(models)

    # Recopilar todos los valores reales para calcular el rango del eje Y
    all_vals = [results[m]["test"].get(metric, 0)
                for m in models for metric in metrics]
    v_min, v_max = min(all_vals), max(all_vals)
    # Margen: 5% del rango observado, con un piso de 0.02 para no
    # comprimir demasiado cuando todos los valores son muy similares
    margin = max((v_max - v_min) * 0.08, 0.02)
    y_low  = max(0.0, v_min - margin)
    y_high = min(1.05, v_max + margin * 2)  # más margen arriba para las etiquetas

    x     = range(len(metrics))
    width = 0.8 / n_models  # el grupo completo ocupa 80% del espacio entre ticks
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (model_name, label) in enumerate(zip(models, labels)):
        vals = [results[model_name]["test"].get(m, 0) for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label=label,
                      color=COLORS.get(model_name, "gray"),
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (y_high - y_low) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90 if n_models > 4 else 0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score"])
    ax.set_ylim(y_low, y_high)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas en test set")
    ax.legend(loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.32))
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()

    path = output_dir / "test_comparison_barplot.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",        type=Path, required=True,
                        help="Archivo de log generado por run_experiments.py")
    parser.add_argument("--output_dir", type=Path, default=Path("./results/curves"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nParsando log: {args.log}")
    results = parse_log(args.log)

    if not results:
        print("  No se encontraron métricas en el log.")
        return

    print(f"  Modelos encontrados: {list(results.keys())}")
    for model, data in results.items():
        print(f"  {model}: {len(data['epochs'])} épocas  |  "
              f"test F1={data['test'].get('f1', 'N/A')}")

    print("\nExportando CSVs...")
    export_csvs(results, args.output_dir)

    print("\nGenerando figuras...")
    plot_metric_comparison(results, "val_f1",
                           "F1-score (validación)",
                           "Evolución del F1-score en validación",
                           args.output_dir / "val_f1_comparison.png")

    plot_metric_comparison(results, "val_loss",
                           "Val Loss",
                           "Evolución de la pérdida de validación",
                           args.output_dir / "val_loss_comparison.png")

    plot_metric_comparison(results, "loss",
                           "Train Loss",
                           "Evolución de la pérdida de entrenamiento",
                           args.output_dir / "train_loss_comparison.png")

    plot_per_model(results, args.output_dir)
    plot_test_comparison(results, args.output_dir)

    print(f"\nTodo guardado en {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()