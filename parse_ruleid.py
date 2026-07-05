import re
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

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

COLORS = {
    "transformer_logkey":        "#2B6CB0",
    "deeplog_logkey":            "#276749",
    "logformer_logkey_pretrain": "#553C9A",
    "logformer_logkey_tuned":    "#975A16",
}
LABELS = {
    "transformer_logkey":        "TimeAwareTransformer-LogKey",
    "deeplog_logkey":            "DeepLog-LogKey",
    "logformer_logkey_pretrain": "LogFormer-LogKey (pre-train)",
    "logformer_logkey_tuned":    "LogFormer-LogKey (adapter tuning)",
}

def parse_log(log_path):
    results = {}
    current = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MODEL_HEADER.search(line)
            if m:
                current = m.group(1).lower()
                results[current] = {"epochs": [], "test": {}}
                continue
            if current is None:
                continue
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
            m = TEST_LINE.search(line)
            if m:
                results[current]["test"] = {
                    "accuracy":  float(m.group(1)),
                    "precision": float(m.group(2)),
                    "recall":    float(m.group(3)),
                    "f1":        float(m.group(4)),
                }
    return results

def plot_comparison(results, metric, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        epochs_with_metric = [e for e in data["epochs"] if metric in e]
        if not epochs_with_metric:
            continue
        epochs = [e["epoch"] for e in epochs_with_metric]
        values = [e[metric]  for e in epochs_with_metric]
        ax.plot(epochs, values, label=LABELS.get(name, name),
                color=COLORS.get(name, "gray"),
                linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {Path(out_path).name}")

def plot_per_model(results, out_dir):
    for name, data in results.items():
        epochs_full = [e for e in data["epochs"]
                       if "val_f1" in e and "loss" in e and "val_loss" in e]
        if not epochs_full:
            print(f"  [SKIP] {name}: sin datos suficientes")
            continue
        epochs     = [e["epoch"]    for e in epochs_full]
        losses     = [e["loss"]     for e in epochs_full]
        val_losses = [e["val_loss"] for e in epochs_full]
        val_f1s    = [e["val_f1"]  for e in epochs_full]
        val_rec    = [e["val_rec"] for e in epochs_full]
        val_pre    = [e["val_prec"] for e in epochs_full]

        color = COLORS.get(name, "gray")
        label = LABELS.get(name, name)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Loss", color="gray")
        ax1.plot(epochs, losses,     color="gray",  linewidth=1.5,
                 linestyle="--", label="Train Loss")
        ax1.plot(epochs, val_losses, color="black", linewidth=1.5,
                 linestyle=":",  label="Val Loss")
        ax1.tick_params(axis="y", labelcolor="gray")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Métricas de validación")
        ax2.plot(epochs, val_f1s, color=color, linewidth=2,
                 marker="o", markersize=4, label="Val F1")
        ax2.plot(epochs, val_rec, color=color, linewidth=1.2,
                 linestyle=":",  label="Val Recall")
        ax2.plot(epochs, val_pre, color=color, linewidth=1.2,
                 linestyle="-.", label="Val Precision")
        ax2.set_ylim(0.5, 1.02)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        ax1.set_title(f"Curvas de entrenamiento — {label}")
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        path = Path(out_dir) / f"{name}_training_detail.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {path.name}")

def export_csvs(results, out_dir):
    fields = ["epoch","loss","val_loss","val_acc","val_prec","val_rec","val_f1"]
    for name, data in results.items():
        if not data["epochs"]:
            continue
        path = Path(out_dir) / f"{name}_training_curves.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data["epochs"])
        print(f"  [OK] {path.name}")

# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path",      type=Path, required=True)
    parser.add_argument("--output_dir",   type=Path, default=Path("./results_ruleid/curves"))

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path
    out_dir  = args.output_dir
    Path(out_dir).mkdir(exist_ok=True)

    print("Parseando log...")
    results = parse_log(log_path)
    for name, data in results.items():
        print(f"  {name}: {len(data['epochs'])} épocas | "
            f"test F1={data['test'].get('f1','N/A')}")

    print("\nExportando CSVs...")
    export_csvs(results, out_dir)

    print("\nGenerando curvas comparativas...")
    plot_comparison(results, "val_f1",
                    "F1-score (validación)",
                    "Evolución del F1-score — representación rule_id-only",
                    f"{out_dir}/val_f1_comparison_ruleid.png")

    plot_comparison(results, "val_loss",
                    "Val Loss",
                    "Evolución de la pérdida de validación — representación rule_id-only",
                    f"{out_dir}/val_loss_comparison_ruleid.png")

    plot_comparison(results, "loss",
                    "Train Loss",
                    "Evolución de la pérdida de entrenamiento — representación rule_id-only",
                    f"{out_dir}/train_loss_comparison_ruleid.png")

    print("\nGenerando curvas individuales...")
    plot_per_model(results, out_dir)

    print(f"\nListo. Figuras en {out_dir}")
    
if __name__ == "__main__":
    main()