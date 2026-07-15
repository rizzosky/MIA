"""
EDA - Detección de Anomalías en Logs de SOC
============================================
Uso:
    python eda_soc_logs.py \
        --normal  /ruta/a/carpeta/Normal \
        --incidente /ruta/a/carpeta/Incidente \
        --salida  /ruta/a/carpeta/salida_eda
 
Genera:
    - Tablas en CSV (para incluir en LaTeX)
    - Figuras en PNG/PDF (para incluir en el documento)
    - Reporte resumen en texto
"""
 
import os
import json
import argparse
import warnings
from pathlib import Path
from collections import Counter
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
 
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})
 
# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
 
def cargar_jsonl(carpeta: Path, etiqueta: int) -> pd.DataFrame:
    """Lee todos los archivos JSONL de una carpeta y devuelve un DataFrame."""
    registros = []
    archivos = sorted(carpeta.glob("*.json"))
    print(f"  Cargando {len(archivos)} archivo(s) de '{carpeta.name}'...")
    for archivo in archivos:
        with open(archivo, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    obj = json.loads(linea)
                    obj["_label"] = etiqueta          # 0=normal, 1=anomalo
                    obj["_archivo"] = archivo.name
                    registros.append(obj)
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
 
 
def cargar_todo(carpeta_normal: Path, carpeta_incidente: Path) -> pd.DataFrame:
    df_n = cargar_jsonl(carpeta_normal,    etiqueta=0)
    df_i = cargar_jsonl(carpeta_incidente, etiqueta=1)
    df   = pd.concat([df_n, df_i], ignore_index=True)
    df["_label_str"] = df["_label"].map({0: "Normal", 1: "Anómalo"})
    return df
 
 
# ─────────────────────────────────────────────
# 2. ESTADÍSTICAS GENERALES
# ─────────────────────────────────────────────
 
def estadisticas_generales(df: pd.DataFrame, salida: Path):
    resumen = []
    for etiqueta, nombre in [(0, "Normal"), (1, "Anómalo")]:
        sub = df[df["_label"] == etiqueta]
        if sub.empty:
            continue
        dur = sub["timestamp"].max() - sub["timestamp"].min()
        iai = sub["timestamp"].diff().dt.total_seconds().dropna()
        resumen.append({
            "Conjunto":               nombre,
            "Total de eventos":       f"{len(sub):,}",
            "Agentes distintos":      sub["agent_id"].nunique() if "agent_id" in sub else "N/A",
            "Log keys distintas":     sub["rule_id"].nunique()  if "rule_id"  in sub else "N/A",
            "Primer evento":          str(sub["timestamp"].min()),
            "Último evento":          str(sub["timestamp"].max()),
            "Duración total":         str(dur),
            "Inter-arrival medio (s)": f"{iai.mean():.4f}" if not iai.empty else "N/A",
            "Inter-arrival mediana (s)": f"{iai.median():.4f}" if not iai.empty else "N/A",
        })
 
    df_res = pd.DataFrame(resumen).T
    df_res.columns = df_res.iloc[0]
    df_res = df_res[1:]
    csv_path = salida / "tabla_estadisticas_generales.csv"
    df_res.to_csv(csv_path)
    print(f"  [OK] {csv_path.name}")
 
    # También imprime en pantalla
    print("\n── Estadísticas generales ──")
    print(df_res.to_string())
    return df_res
 
 
# ─────────────────────────────────────────────
# 3. BALANCE DE CLASES (VENTANAS)
# ─────────────────────────────────────────────
 
def balance_ventanas(df: pd.DataFrame, salida: Path,
                     window_min: int = 5, step_min: int = 1):
    """
    Construye ventanas deslizantes y calcula el balance de clases.
    Una ventana es anómala si contiene AL MENOS un evento anómalo.
    """
    print("  Construyendo ventanas temporales...")
    df_sorted = df.sort_values("timestamp").copy()
    t_min = df_sorted["timestamp"].min()
    t_max = df_sorted["timestamp"].max()
 
    step   = pd.Timedelta(minutes=step_min)
    window = pd.Timedelta(minutes=window_min)
 
    starts  = pd.date_range(t_min, t_max - window, freq=step)
    labels  = []
    n_evts  = []
 
    for start in starts:
        end  = start + window
        mask = (df_sorted["timestamp"] >= start) & (df_sorted["timestamp"] < end)
        sub  = df_sorted[mask]
        labels.append(int(sub["_label"].max()) if not sub.empty else 0)
        n_evts.append(len(sub))
 
    df_vent = pd.DataFrame({"start": starts, "label": labels, "n_eventos": n_evts})
    df_vent = df_vent[df_vent["n_eventos"] > 0]   # descarta ventanas vacías
 
    conteo = df_vent["label"].value_counts().rename({0: "Normal", 1: "Anómalo"})
    pct    = (conteo / conteo.sum() * 100).round(2)
 
    df_bal = pd.DataFrame({"Ventanas": conteo, "Porcentaje (%)": pct})
    csv_path = salida / "tabla_balance_clases.csv"
    df_bal.to_csv(csv_path)
    print(f"  [OK] {csv_path.name}")
    print(f"\n── Balance de clases (W={window_min}min, step={step_min}min) ──")
    print(df_bal.to_string())
 
    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 
    axes[0].bar(conteo.index, conteo.values,
                color=["#4C9BE8", "#E8614C"], edgecolor="white", linewidth=0.8)
    axes[0].set_title("Cantidad de ventanas por clase")
    axes[0].set_ylabel("Número de ventanas")
    for i, v in enumerate(conteo.values):
        axes[0].text(i, v + conteo.max()*0.01, f"{v:,}", ha="center", fontsize=10)
 
    axes[1].pie(conteo.values, labels=conteo.index,
                colors=["#4C9BE8", "#E8614C"],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    axes[1].set_title("Distribución porcentual")
 
    fig.suptitle(f"Balance de clases — ventanas de {window_min} min (paso {step_min} min)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig_path = salida / "fig_balance_clases.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fig_path.name}")
 
    return df_vent
 
 
# ─────────────────────────────────────────────
# 4. DISTRIBUCIÓN DE EVENTOS POR VENTANA
# ─────────────────────────────────────────────
 
def dist_eventos_por_ventana(df_vent: pd.DataFrame, salida: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {0: "#4C9BE8", 1: "#E8614C"}
    for lbl, nombre in [(0, "Normal"), (1, "Anómalo")]:
        sub = df_vent[df_vent["label"] == lbl]["n_eventos"]
        if sub.empty:
            continue
        ax.hist(sub, bins=60, alpha=0.7, label=nombre,
                color=colors[lbl], edgecolor="white", linewidth=0.4)
 
    ax.set_xlabel("Número de eventos por ventana")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de eventos por ventana de tiempo")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig_path = salida / "fig_dist_eventos_por_ventana.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fig_path.name}")
 
    # Estadísticas descriptivas (con percentiles altos, usados para
    # justificar la elección de max_seq_len en el documento)
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    desc = (df_vent.groupby("label")["n_eventos"]
                   .describe(percentiles=percentiles).round(2))
    desc.index = desc.index.map({0: "Normal", 1: "Anómalo"})
 
    # Fila adicional: ambas clases combinadas
    combinado = (df_vent["n_eventos"]
                 .describe(percentiles=percentiles).round(2))
    combinado.name = "Combinado"
    desc = pd.concat([desc, combinado.to_frame().T])
 
    csv_path = salida / "tabla_eventos_por_ventana.csv"
    desc.to_csv(csv_path)
    print(f"  [OK] {csv_path.name}")
    print("\n── Eventos por ventana ──")
    print(desc.to_string())
 
    # Valores explícitos citados en el documento
    p95_comb = float(np.percentile(df_vent["n_eventos"], 95))
    p99_comb = float(np.percentile(df_vent["n_eventos"], 99))
    print(f"\n  P95 combinado (ambas clases): {p95_comb:.1f} eventos/ventana")
    print(f"  P99 combinado (ambas clases): {p99_comb:.1f} eventos/ventana")
 
 
# ─────────────────────────────────────────────
# 5. TOP-N LOG KEYS POR CONJUNTO
# ─────────────────────────────────────────────
 
def top_log_keys(df: pd.DataFrame, salida: Path, top_n: int = 20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    tablas = {}
 
    for ax, (etiqueta, nombre, color) in zip(
        axes,
        [(0, "Normal", "#4C9BE8"), (1, "Anómalo", "#E8614C")]
    ):
        sub = df[df["_label"] == etiqueta]
        if sub.empty or "rule_id" not in sub:
            continue
 
        conteo = sub["rule_id"].value_counts().head(top_n)
 
        # Intentar agregar descripción si está disponible
        if "rule_description" in sub.columns:
            desc_map = (sub.dropna(subset=["rule_description"])
                          .drop_duplicates("rule_id")
                          .set_index("rule_id")["rule_description"])
            etiquetas = [f"{rid}: {desc_map.get(rid, '')[:35]}"
                         for rid in conteo.index]
        else:
            etiquetas = conteo.index.astype(str).tolist()
 
        ax.barh(etiquetas[::-1], conteo.values[::-1],
                color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Top {top_n} log keys — {nombre}")
        ax.set_xlabel("Apariciones")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
        df_tabla = pd.DataFrame({
            "rule_id":      conteo.index,
            "descripcion":  [desc_map.get(r, "") if "rule_description" in sub.columns
                             else "" for r in conteo.index],
            "apariciones":  conteo.values,
        })
        tablas[nombre] = df_tabla
        csv_path = salida / f"tabla_top{top_n}_logkeys_{nombre.lower()}.csv"
        df_tabla.to_csv(csv_path, index=False)
        print(f"  [OK] {csv_path.name}")
 
    fig.tight_layout()
    fig_path = salida / f"fig_top{top_n}_logkeys.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fig_path.name}")
    return tablas
 
 
# ─────────────────────────────────────────────
# 6. VOLUMEN DE EVENTOS EN EL TIEMPO
# ─────────────────────────────────────────────
 
def volumen_en_tiempo(df: pd.DataFrame, salida: Path, freq: str = "1h"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
 
    for ax, (etiqueta, nombre, color) in zip(
        axes,
        [(0, "Normal", "#4C9BE8"), (1, "Anómalo", "#E8614C")]
    ):
        sub = df[df["_label"] == etiqueta].copy()
        if sub.empty:
            continue
        serie = sub.set_index("timestamp").resample(freq).size()
        ax.plot(serie.index, serie.values, color=color, linewidth=0.9, alpha=0.85)
        ax.fill_between(serie.index, serie.values, alpha=0.15, color=color)
        ax.set_title(f"Volumen de eventos por hora — {nombre}")
        ax.set_ylabel("Eventos")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
    fig.tight_layout()
    fig_path = salida / "fig_volumen_en_tiempo.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fig_path.name}")
 
 
# ─────────────────────────────────────────────
# 7. DISTRIBUCIÓN DE TÁCTICAS MITRE
# ─────────────────────────────────────────────
 
def distribucion_mitre(df: pd.DataFrame, salida: Path):
    if "mitre_tactic" not in df.columns:
        print("  [SKIP] Campo mitre_tactic no encontrado.")
        return
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
    for ax, (etiqueta, nombre, color) in zip(
        axes,
        [(0, "Normal", "#4C9BE8"), (1, "Anómalo", "#E8614C")]
    ):
        sub = df[df["_label"] == etiqueta]["mitre_tactic"].dropna()
        # Puede ser lista o string
        tacticas = []
        for val in sub:
            if isinstance(val, list):
                tacticas.extend(val)
            elif isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    tacticas.extend(parsed if isinstance(parsed, list) else [parsed])
                except Exception:
                    tacticas.append(val)
 
        if not tacticas:
            ax.set_visible(False)
            continue
 
        conteo = Counter(tacticas).most_common(15)
        keys   = [c[0] for c in conteo][::-1]
        vals   = [c[1] for c in conteo][::-1]
 
        ax.barh(keys, vals, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Tácticas MITRE ATT&CK — {nombre}")
        ax.set_xlabel("Apariciones")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
        csv_path = salida / f"tabla_mitre_tacticas_{nombre.lower()}.csv"
        pd.DataFrame({"tactica": keys[::-1], "apariciones": vals[::-1]}).to_csv(
            csv_path, index=False)
        print(f"  [OK] {csv_path.name}")
 
    fig.tight_layout()
    fig_path = salida / "fig_mitre_tacticas.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fig_path.name}")
 
 
# ─────────────────────────────────────────────
# 8. DISTRIBUCIÓN DE NIVEL DE ALERTA
# ─────────────────────────────────────────────
 
def distribucion_nivel(df: pd.DataFrame, salida: Path):
    if "rule_level" not in df.columns:
        return
 
    fig, ax = plt.subplots(figsize=(10, 5))
    niveles = sorted(df["rule_level"].dropna().unique())
    width   = 0.35
    x       = np.arange(len(niveles))
 
    for i, (etiqueta, nombre, color) in enumerate(
        [(0, "Normal", "#4C9BE8"), (1, "Anómalo", "#E8614C")]
    ):
        sub    = df[df["_label"] == etiqueta]
        conteo = sub["rule_level"].value_counts().reindex(niveles, fill_value=0)
        ax.bar(x + i * width, conteo.values, width, label=nombre,
               color=color, edgecolor="white", linewidth=0.6)
 
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(int(n)) for n in niveles])
    ax.set_xlabel("Nivel de alerta Wazuh (rule_level)")
    ax.set_ylabel("Número de eventos")
    ax.set_title("Distribución de niveles de alerta por clase")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig_path = salida / "fig_distribucion_nivel_alerta.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fig_path.name}")
 
 
# ─────────────────────────────────────────────
# 9. INTER-ARRIVAL TIME
# ─────────────────────────────────────────────
 
def inter_arrival(df: pd.DataFrame, salida: Path, percentil: float = 99.0):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    for ax, (etiqueta, nombre, color) in zip(
        axes,
        [(0, "Normal", "#4C9BE8"), (1, "Anómalo", "#E8614C")]
    ):
        sub = df[df["_label"] == etiqueta].sort_values("timestamp")
        iai = sub["timestamp"].diff().dt.total_seconds().dropna()
        iai = iai[iai >= 0]
        cap = np.percentile(iai, percentil)
        iai_cap = iai[iai <= cap]
 
        ax.hist(iai_cap, bins=80, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.set_title(f"Inter-arrival time — {nombre}\n(p{int(percentil)} = {cap:.2f}s)")
        ax.set_xlabel("Segundos entre eventos consecutivos")
        ax.set_ylabel("Frecuencia")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
    fig.tight_layout()
    fig_path = salida / "fig_inter_arrival.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fig_path.name}")
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="EDA - Logs SOC")
    parser.add_argument("--normal",    type=Path, required=True,
                        help="Carpeta con archivos JSONL normales")
    parser.add_argument("--incidente", type=Path, required=True,
                        help="Carpeta con archivos JSONL de incidente")
    parser.add_argument("--salida",    type=Path, default=Path("./eda_output"),
                        help="Carpeta de salida para figuras y tablas")
    parser.add_argument("--window",    type=int, default=5,
                        help="Tamaño de ventana en minutos (default: 5)")
    parser.add_argument("--step",      type=int, default=1,
                        help="Paso entre ventanas en minutos (default: 1)")
    parser.add_argument("--topn",      type=int, default=20,
                        help="Top N log keys a mostrar (default: 20)")
    args = parser.parse_args()
 
    args.salida.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  EDA - Detección de anomalías en logs de SOC")
    print(f"{'='*55}")
 
    print("\n[1/8] Cargando datos...")
    df = cargar_todo(args.normal, args.incidente)
    print(f"  Total de eventos cargados: {len(df):,}")
 
    print("\n[2/8] Estadísticas generales...")
    estadisticas_generales(df, args.salida)
 
    print("\n[3/8] Balance de clases (ventanas)...")
    df_vent = balance_ventanas(df, args.salida, args.window, args.step)
 
    print("\n[4/8] Distribución de eventos por ventana...")
    dist_eventos_por_ventana(df_vent, args.salida)
 
    print("\n[5/8] Top log keys por conjunto...")
    top_log_keys(df, args.salida, args.topn)
 
    print("\n[6/8] Volumen de eventos en el tiempo...")
    volumen_en_tiempo(df, args.salida)
 
    print("\n[7/8] Tácticas MITRE ATT&CK...")
    distribucion_mitre(df, args.salida)
 
    print("\n[8/8] Nivel de alerta y inter-arrival time...")
    distribucion_nivel(df, args.salida)
    inter_arrival(df, args.salida)
 
    print(f"\n{'='*55}")
    print(f"  EDA completo. Salida en: {args.salida.resolve()}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()