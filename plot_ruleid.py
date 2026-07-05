"""
plot_ruleid.py
---------------
Genera los graficos de barras comparativos de metricas en test set,
leyendo directamente de los archivos JSON de resultados en lugar de
tener los valores hardcodeados. Reproducible ante nuevos experimentos.

Uso:
    python plot_ruleid.py \
        --results_dir       ./results \
        --results_ruleid_dir ./results_ruleid \
        --output_dir        ./results/curves
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

METRICS = ["accuracy", "precision", "recall", "f1"]


def _extract_metrics(d):
    return {m: d[m] for m in METRICS if m in d}


def load_rich_representation_results(results_dir):
    results = {}
    simple_models = {
        "transformer": "transformer_metrics.json",
        "bert":        "bert_metrics.json",
        "deeplog":     "deeplog_metrics.json",
    }
    for key, fname in simple_models.items():
        path = results_dir / fname
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            results[key] = _extract_metrics(d)
        else:
            print(f"  [WARN] No encontrado: {path}")

    logformer_path = results_dir / "logformer_metrics.json"
    if logformer_path.exists():
        with open(logformer_path) as f:
            d = json.load(f)
        if "stage1_pretraining" in d:
            results["logformer_pretrain"] = _extract_metrics(d["stage1_pretraining"])
            results["logformer_tuned"]    = _extract_metrics(d["stage2_adapter_tuning"])
    else:
        print(f"  [WARN] No encontrado: {logformer_path}")

    return results


def load_ruleid_representation_results(results_dir):
    results = {}
    comparison_path = results_dir / "comparison_results_ruleid.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            d = json.load(f)
        for key, metrics in d.get("results", {}).items():
            results[key] = _extract_metrics(metrics)
    else:
        for key in ["transformer_logkey", "deeplog_logkey"]:
            path = results_dir / f"{key}_metrics.json"
            if path.exists():
                with open(path) as f:
                    results[key] = _extract_metrics(json.load(f))
            else:
                print(f"  [WARN] No encontrado: {path}")

    logformer_path = results_dir / "logformer_logkey_metrics.json"
    if logformer_path.exists():
        with open(logformer_path) as f:
            d = json.load(f)
        if "stage1_pretraining" in d:
            results["logformer_logkey_pretrain"] = _extract_metrics(d["stage1_pretraining"])
            results["logformer_logkey_tuned"]    = _extract_metrics(d["stage2_adapter_tuning"])
    else:
        print(f"  [WARN] No encontrado: {logformer_path}")

    return results


COLORS_RICH = {
    "transformer":        "#4C9BE8",
    "bert":               "#E8614C",
    "logformer_pretrain": "#9B6FD8",
    "logformer_tuned":    "#D89B2E",
    "deeplog":            "#4CAF50",
}
LABELS_RICH = {
    "transformer":        "TimeAwareTransformer",
    "bert":               "TimeAwareBERT",
    "logformer_pretrain": "LogFormer (pre-train)",
    "logformer_tuned":    "LogFormer (adapter tuning)",
    "deeplog":            "DeepLog Baseline",
}

COLORS_RULEID = {
    "transformer_logkey":        "#2B6CB0",
    "deeplog_logkey":            "#276749",
    "logformer_logkey_pretrain": "#553C9A",
    "logformer_logkey_tuned":    "#975A16",
}
LABELS_RULEID = {
    "transformer_logkey":        "TimeAwareTransformer-LogKey",
    "deeplog_logkey":            "DeepLog-LogKey",
    "logformer_logkey_pretrain": "LogFormer-LogKey (pre-train)",
    "logformer_logkey_tuned":    "LogFormer-LogKey (adapter tuning)",
}

ORDER_RICH   = ["transformer", "bert", "logformer_pretrain",
                "logformer_tuned", "deeplog"]
ORDER_RULEID = ["transformer_logkey", "deeplog_logkey",
                "logformer_logkey_pretrain", "logformer_logkey_tuned"]


def _ordered_keys(results, preferred_order):
    ordered = [k for k in preferred_order if k in results]
    extra   = [k for k in results if k not in ordered]
    return ordered + extra


def plot_barchart(results, colors, labels, title, output_path,
                  figsize=(13, 5.5), ylim=(0.93, 1.03),
                  legend_ncol=2, rotation=90):
    models   = list(results.keys())
    n_models = len(models)
    if n_models == 0:
        print(f"  [SKIP] Sin datos para '{title}'")
        return

    x     = range(len(METRICS))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(models):
        vals   = [results[name].get(m, 0) for m in METRICS]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x], vals, width,
            label=labels.get(name, name),
            color=colors.get(name, "gray"),
            edgecolor="white", linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0015,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7.5 if n_models <= 5 else 6,
                rotation=rotation,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score"])
    ax.set_ylim(*ylim)
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.legend(loc="lower center", ncol=legend_ncol, fontsize=9,
              bbox_to_anchor=(0.5, -0.30 if n_models <= 5 else -0.38))
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",        type=Path, default=Path("./results"))
    parser.add_argument("--results_ruleid_dir", type=Path, default=Path("./results_ruleid"))
    parser.add_argument("--output_dir",         type=Path, default=Path("./results/curves"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando resultados - representacion texto + BERT...")
    rich_results = load_rich_representation_results(args.results_dir)
    print(f"  Modelos encontrados: {list(rich_results.keys())}")

    print("\nCargando resultados - representacion rule_id...")
    ruleid_results = load_ruleid_representation_results(args.results_ruleid_dir)
    print(f"  Modelos encontrados: {list(ruleid_results.keys())}")

    print("\nGenerando barplot rule_id...")
    ordered_ruleid = {k: ruleid_results[k]
                      for k in _ordered_keys(ruleid_results, ORDER_RULEID)}
    plot_barchart(
        ordered_ruleid, COLORS_RULEID, LABELS_RULEID,
        title="Comparacion de metricas en test set - representacion rule_id-only",
        output_path=args.output_dir / "test_comparison_barplot_ruleid.png",
        figsize=(13, 5.5), legend_ncol=2,
    )

    print("\nGenerando barplot representacion rica...")
    ordered_rich = {k: rich_results[k]
                    for k in _ordered_keys(rich_results, ORDER_RICH)}
    plot_barchart(
        ordered_rich, COLORS_RICH, LABELS_RICH,
        title="Comparacion de metricas en test set - texto enriquecido + BERT",
        output_path=args.output_dir / "test_comparison_barplot.png",
        figsize=(12, 5.5), legend_ncol=3,
    )

    print("\nGenerando barplot combinado...")
    all_results = {**ordered_rich, **ordered_ruleid}
    all_colors  = {**COLORS_RICH, **COLORS_RULEID}
    all_labels  = {**LABELS_RICH, **LABELS_RULEID}
    plot_barchart(
        all_results, all_colors, all_labels,
        title="Comparacion completa - todos los modelos y representaciones",
        output_path=args.output_dir / "test_comparison_barplot_all.png",
        figsize=(18, 6), legend_ncol=3,
    )

    print(f"\nListo. Figuras guardadas en {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()