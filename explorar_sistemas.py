#!/usr/bin/env python3
"""
explorar_sistemas.py
--------------------
Explora cuántos agentes, ubicaciones y organizaciones distintas
hay en los archivos JSONL de Wazuh, sin cargar todo en memoria.

Uso:
    python explorar_sistemas.py --carpeta /ruta/carpeta [--limite 500000]

Los archivos pueden estar comprimidos (.xz, .gz) o sin comprimir (.json).
"""

import os
import json
import gzip
import lzma
import argparse
from pathlib import Path
from collections import Counter, defaultdict

def abrir_archivo(path: Path):
    """Abre el archivo independientemente de la compresión."""
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    elif path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        return open(path, "r", encoding="utf-8", errors="replace")

def explorar(carpeta: Path, limite: int):
    archivos = sorted(
        [p for p in carpeta.rglob("*")
         if p.suffix in (".json", ".xz", ".gz") and p.is_file()]
    )
    print(f"Archivos encontrados: {len(archivos)}")

    agentes          = Counter()   # agent_id -> count
    ubicaciones      = Counter()   # location -> count
    grupos           = Counter()   # agent.labels.Group -> count
    pares            = Counter()   # (agent_id, location) -> count
    total            = 0

    for archivo in archivos:
        print(f"  Procesando {archivo.name}...")
        try:
            with abrir_archivo(archivo) as f:
                for linea in f:
                    if total >= limite:
                        break
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        obj = json.loads(linea)
                    except json.JSONDecodeError:
                        continue

                    aid  = obj.get("agent", {}).get("id") or obj.get("agent_id", "?")
                    loc  = obj.get("location") or "?"
                    grp  = (obj.get("agent", {})
                               .get("labels", {})
                               .get("Group", "?"))

                    agentes[aid]   += 1
                    ubicaciones[loc] += 1
                    grupos[grp]    += 1
                    pares[(aid, loc)] += 1
                    total          += 1

        except Exception as e:
            print(f"    [ERROR] {e}")

    print(f"\n{'='*55}")
    print(f"  Total de eventos procesados : {total:,}")
    print(f"  Agentes distintos           : {len(agentes):,}")
    print(f"  Ubicaciones distintas       : {len(ubicaciones):,}")
    print(f"  Organizaciones (Group)      : {len(grupos):,}")
    print(f"  Pares (agente, ubicación)   : {len(pares):,}")
    print(f"{'='*55}")

    print(f"\n── Top 20 agentes por volumen ──")
    for aid, cnt in agentes.most_common(20):
        print(f"  {aid:>10}  {cnt:>10,}")

    print(f"\n── Top 20 ubicaciones por volumen ──")
    for loc, cnt in ubicaciones.most_common(20):
        print(f"  {loc:<40}  {cnt:>8,}")

    print(f"\n── Organizaciones (Groups) ──")
    for grp, cnt in grupos.most_common():
        print(f"  {grp:<40}  {cnt:>8,}")

    print(f"\n── Top 30 pares (agente, ubicación) ──")
    for (aid, loc), cnt in pares.most_common(30):
        print(f"  agente={aid:<8}  loc={loc:<35}  eventos={cnt:>8,}")

    # Guardar CSV con todos los pares
    salida = carpeta / "resumen_sistemas.csv"
    with open(salida, "w", encoding="utf-8") as f:
        f.write("agent_id,location,eventos\n")
        for (aid, loc), cnt in pares.most_common():
            f.write(f"{aid},{loc},{cnt}\n")
    print(f"\n  Resumen guardado en: {salida}")


def main():
    parser = argparse.ArgumentParser(description="Explorar sistemas en logs Wazuh")
    parser.add_argument("--carpeta", type=Path, required=True,
                        help="Carpeta con los archivos JSONL (comprimidos o no)")
    parser.add_argument("--limite",  type=int, default=500_000,
                        help="Máximo de eventos a procesar (default: 500000)")
    args = parser.parse_args()
    explorar(args.carpeta, args.limite)

if __name__ == "__main__":
    main()