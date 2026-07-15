"""
Estimación del submuestreo mínimo aplicable según la memoria disponible
=======================================================================
Modelo de capacidad: el dataset de ventanas almacena, por cada evento,
un embedding de dimensión d en float32, replicado en cada ventana
solapada que lo contiene (window/step ventanas). El tamaño total escala
linealmente con el número de eventos:

    tamaño(N) ≈ N · d · bytes_dtype · (window/step) · overhead

El coeficiente real (bytes/evento) puede medirse directamente desde un
.pkl existente con --pkl y --eventos-pkl, lo que captura además el
overhead de metadatos y serialización.

Uso:
    # Con coeficiente medido desde un dataset real (recomendado):
    python estimar_submuestreo.py \
        --pkl data/windows_pfsense.pkl --eventos-pkl 100000 \
        --eventos-total 6125736 --ram-gb 24

    # Sin pkl (usa el modelo teórico):
    python estimar_submuestreo.py --eventos-total 6125736 --ram-gb 24
"""

import argparse
import os

GB = 1024 ** 3
MB = 1024 ** 2


def coeficiente_teorico(dim: int, bytes_dtype: int, window: int,
                        step: int, overhead: float) -> float:
    """Bytes por evento según el modelo teórico."""
    return dim * bytes_dtype * (window / step) * overhead


def main():
    p = argparse.ArgumentParser(
        description="Estimación de submuestreo mínimo por memoria")
    p.add_argument("--pkl", type=str, default=None,
                   help="Dataset .pkl existente para medir bytes/evento")
    p.add_argument("--eventos-pkl", type=int, default=None,
                   help="Cantidad de eventos contenidos en ese .pkl")
    p.add_argument("--eventos-total", type=int, required=True,
                   help="Eventos del dataset completo (sin submuestrear)")
    p.add_argument("--ram-gb", type=float, default=24.0,
                   help="Memoria unificada total del equipo (GB)")
    p.add_argument("--dim", type=int, default=768,
                   help="Dimensión del embedding (default: 768)")
    p.add_argument("--window", type=int, default=5,
                   help="Tamaño de ventana en minutos")
    p.add_argument("--step", type=int, default=1,
                   help="Paso entre ventanas en minutos")
    p.add_argument("--overhead", type=float, default=1.05,
                   help="Factor de overhead de serialización (teórico)")
    args = p.parse_args()

    # ── Coeficiente bytes/evento ──────────────────────────────────
    if args.pkl and args.eventos_pkl:
        tam = os.path.getsize(args.pkl)
        bpe = tam / args.eventos_pkl
        origen = (f"medido desde {os.path.basename(args.pkl)} "
                  f"({tam / MB:,.1f} MB / {args.eventos_pkl:,} eventos)")
    else:
        bpe = coeficiente_teorico(args.dim, 4, args.window,
                                  args.step, args.overhead)
        origen = (f"teórico: {args.dim} dims × 4 B (float32) × "
                  f"{args.window}/{args.step} ventanas solapadas × "
                  f"{args.overhead} overhead")

    print("=" * 64)
    print("  Estimación de submuestreo mínimo por memoria")
    print("=" * 64)
    print(f"  Coeficiente: {bpe:,.0f} bytes/evento ({origen})")
    print(f"  Eventos totales del dataset: {args.eventos_total:,}")
    print(f"  Dataset completo ocuparía:   "
          f"{args.eventos_total * bpe / GB:,.1f} GB")
    print(f"  Memoria unificada total:     {args.ram_gb:.0f} GB")
    print()

    # ── Presupuestos de memoria ───────────────────────────────────
    # La memoria utilizable es menor que la total: el SO, el modelo,
    # las activaciones y el pico transitorio de deserialización del
    # pickle (que puede duplicar el residente durante la carga)
    # consumen parte del total. Se evalúan tres escenarios.
    escenarios = [
        ("Conservador (33% de la RAM: pico de carga 2x + modelo)",
         args.ram_gb / 3),
        ("Moderado (50% de la RAM)", args.ram_gb / 2),
        ("Límite práctico (66% de la RAM)", args.ram_gb * 2 / 3),
    ]

    print(f"  {'Escenario':<55} {'Budget':>7} {'Max eventos':>12} "
          f"{'% del total':>11} {'Submuestreo':>11}")
    print("  " + "-" * 100)
    for nombre, budget_gb in escenarios:
        n_max = int(budget_gb * GB / bpe)
        pct = 100 * n_max / args.eventos_total
        factor = args.eventos_total / n_max if n_max else float("inf")
        print(f"  {nombre:<55} {budget_gb:>5.1f}GB {n_max:>12,} "
              f"{pct:>10.1f}% {'1 de ' + f'{factor:,.1f}':>11}")

    print()
    # ── Alternativas para elevar el techo ─────────────────────────
    n_mod = int((args.ram_gb / 2) * GB / bpe)
    print("  Alternativas para elevar el techo (sobre escenario moderado):")
    print(f"    float16 en lugar de float32 (x2):      "
          f"{n_mod * 2:>12,} eventos")
    print(f"    sin duplicación por solapamiento (x{args.window/args.step:.0f}): "
          f"{n_mod * int(args.window/args.step):>12,} eventos")
    print(f"    ambas combinadas (x{2*args.window/args.step:.0f}):              "
          f"{n_mod * 2 * int(args.window/args.step):>12,} eventos")
    print("    memory-mapping (np.memmap/HDF5): sin límite de RAM")
    print("=" * 64)


if __name__ == "__main__":
    main()