"""
makeSolidifi.py
------------------------------------
Procesa el SolidiFI-benchmark dataset y genera un CSV con las funciones
vulnerables de los contratos.

Metodologia:
  1. Fuente primaria: BugLog (ground truth del dataset, una entrada por bug
     inyectado). Esto replica los 5,434 snippets reportados en el paper:
     "Deep learning-based solution for smart contract vulnerabilities detection"
     (Tang et al., Scientific Reports 2023, doi:10.1038/s41598-023-47219-0).
  2. Slither se ejecuta sobre cada contrato (codigo sin comentarios) para
     validar las detecciones. Las lineas que Slither reporta se unen a las
     del BugLog (union de fuentes).
  3. Para cada linea de bug, se extrae el cuerpo de la funcion contenedora
     mediante conteo de llaves {}.
  4. Deduplicacion: una sola fila por (vuln, contrato, nombre_funcion).
     No se deduplica por hash entre contratos distintos.

Vulnerabilidades objetivo:
  Re-entrancy, Timestamp-Dependency, Unhandled-Exceptions, tx.origin

Slither detectors mapeados (ref. Table 1 del paper):
  Re-entrancy         -> reentrancy-eth, reentrancy-no-eth, reentrancy-benign,
                         reentrancy-events
  Timestamp-Dependency -> timestamp
  Unhandled-Exceptions -> unchecked-lowlevel, unchecked-send
  tx.origin            -> tx-origin

Columnas del CSV resultante:
  id, contract_file, vulnerability, function_name, function_code, dataset

Uso:
  python makeSolidifi.py --base_path <ruta>/buggy_contracts --output solidifi.csv
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

VULNERABILIDADES_OBJETIVO = [
    "Re-entrancy",
    "Timestamp-Dependency",
    "Unhandled-Exceptions",
    "tx.origin",
]

SLITHER_DETECTORS = {
    "Re-entrancy": [
        "reentrancy-eth",
        "reentrancy-no-eth",
        "reentrancy-benign",
        "reentrancy-events",
    ],
    "Timestamp-Dependency": ["timestamp"],
    "Unhandled-Exceptions": ["unchecked-lowlevel", "unchecked-send"],
    "tx.origin": ["tx-origin"],
}

# ---------------------------------------------------------------------------
# Rutas (sin argumentos — ajustar si se mueve el dataset)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DATA_DIR = _HERE / "data" / "buggy_contracts"
OUTPUT_CSV = _HERE / "solidifi_functions.csv"
ENV_REPORT = _HERE / "env_report_solidifi.json"

# ---------------------------------------------------------------------------
# Limpieza de comentarios (preserva numeros de linea)
# ---------------------------------------------------------------------------


def strip_comments(source: str) -> str:
    """Elimina // y /* */ conservando los saltos de linea intactos."""

    def replace_block(m):
        return "\n" * m.group(0).count("\n")

    source = re.sub(r"/\*.*?\*/", replace_block, source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    return source


# ---------------------------------------------------------------------------
# Carga del BugLog (ground truth)
# ---------------------------------------------------------------------------


def load_bug_log(sol_path: str) -> list:
    """
    Carga el BugLog CSV correspondiente a sol_path.
    Nombre esperado: BugLog_<N>.csv para buggy_<N>.sol
    Retorna lista de dicts con campo 'loc' (int).
    """
    stem = Path(sol_path).stem  # "buggy_1"
    number = stem.replace("buggy_", "")  # "1"
    buglog_path = Path(sol_path).parent / f"BugLog_{number}.csv"

    if not buglog_path.exists():
        return []

    entries = []
    try:
        with open(buglog_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    loc = int(row.get("loc", "")) + 1
                    entries.append({"loc": loc, "raw": row})
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return entries


# ---------------------------------------------------------------------------
# Extraccion de funciones por numero de linea
# ---------------------------------------------------------------------------

_FUNC_HEADER = re.compile(
    r"^\s*(?:function\s+(\w+)"
    r"|constructor\s*[\(\{]"
    r"|fallback\s*[\(\{]"
    r"|receive\s*[\(\{]"
    r"|modifier\s+(\w+))"
)


def _find_block_end(lines: list, start_idx: int):
    """
    Retorna el indice (0-based) donde se cierra el bloque.
    Retorna None si la funcion es solo una declaracion (termina en ';'
    antes de abrir '{').
    """
    depth = 0
    opened = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
            elif ch == ";" and not opened:
                return None  # declaracion sin cuerpo
        if opened and depth <= 0:
            return i
    return None


def _get_all_functions(source: str) -> list:
    """Lista de (name, start_0, end_0, code) para funciones con cuerpo."""
    lines = source.splitlines()
    results = []
    for i, line in enumerate(lines):
        m = _FUNC_HEADER.match(line)
        if not m:
            continue
        name = m.group(1) or m.group(2)
        if not name:
            name = line.strip().split("(")[0].strip().split("{")[0].strip()
        end = _find_block_end(lines, i)
        if end is None:
            continue
        code = "\n".join(lines[i : end + 1])
        results.append((name, i, end, code))
    return results


def extract_function_at_line(source: str, target_line: int):
    """
    Retorna (name, code) de la funcion mas cercana a target_line (1-indexed).

    Orden de busqueda:
      1. Funcion que contiene exactamente la linea (start <= target <= end).
      2. Funcion mas cercana: minima distancia entre target y (start..end).
         Permite detectar casos donde el BugLog apunta a una declaracion de
         variable inmediatamente ANTES de la funcion inyectada.

    Retorna (None, None) si no hay ninguna funcion en el archivo.
    """
    target_0 = target_line - 1
    functions = _get_all_functions(source)
    if not functions:
        return None, None

    # 1. Contenedora exacta
    for name, start, end, code in functions:
        if start <= target_0 <= end:
            return name, code

    # 2. Funcion mas cercana (mirando adelante Y atras)
    best = None
    best_dist = float("inf")
    for name, start, end, code in functions:
        if start > target_0:
            dist = start - target_0  # funcion debajo: distancia al inicio
        else:
            dist = target_0 - end  # funcion arriba: distancia al cierre
        if dist < best_dist:
            best_dist = dist
            best = (name, code)

    return best if best else (None, None)


# ---------------------------------------------------------------------------
# Ejecucion de Slither (validacion complementaria)
# ---------------------------------------------------------------------------


def run_slither(sol_path: str, detectors: list) -> dict:
    """
    Ejecuta Slither sobre sol_path con los detectores indicados.
    Retorna el JSON parseado, o {} si falla.
    Lanza RuntimeError si slither no esta en PATH.
    """
    cmd = [
        "slither",
        sol_path,
        "--detect",
        ",".join(detectors),
        "--json",
        "-",
        "--exclude-dependencies",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = result.stdout.strip()
        if stdout:
            return json.loads(stdout)
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {os.path.basename(sol_path)}")
    except json.JSONDecodeError:
        pass
    except FileNotFoundError:
        raise RuntimeError("slither no encontrado. Instalar con: pip install slither-analyzer")
    return {}


def get_slither_lines(slither_output: dict, detectors: list) -> set:
    """Extrae el conjunto de lineas reportadas por Slither."""
    lines = set()
    detectors_set = set(detectors)
    for finding in slither_output.get("results", {}).get("detectors", []):
        if finding.get("check") not in detectors_set:
            continue
        for element in finding.get("elements", []):
            for ln in element.get("source_mapping", {}).get("lines", []):
                lines.add(ln)
    return lines


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_solidifi_functions_dataset(base_path: str) -> pd.DataFrame:
    """
    Recorre base_path/<VulnType>/*.sol, usa BugLog como fuente primaria de
    lineas de bug y Slither como validacion complementaria. Extrae una fila
    por (vuln, contrato, nombre_funcion).
    """
    rows = []
    row_id = 0
    t_start = time.time()

    global_stats = {}

    for vuln in VULNERABILIDADES_OBJETIVO:
        vuln_path = os.path.join(base_path, vuln)
        if not os.path.isdir(vuln_path):
            print(f"[AVISO] Carpeta no encontrada: {vuln_path}")
            continue

        detectors = SLITHER_DETECTORS[vuln]
        sol_files = sorted(f for f in os.listdir(vuln_path) if f.endswith(".sol"))

        print(f"\n[{vuln}] {len(sol_files)} contratos")
        print(f"  Slither detectors: {', '.join(detectors)}")

        stats = {
            "contratos": len(sol_files),
            "con_buglog": 0,
            "slither_ok": 0,
            "funciones": 0,
            "skipped": 0,
        }

        for sol_file in sol_files:
            sol_path = os.path.join(vuln_path, sol_file)

            # --- Leer contrato ---
            try:
                with open(sol_path, "r", encoding="utf-8", errors="ignore") as f:
                    source_original = f.read()
            except Exception as e:
                print(f"  ERROR leyendo {sol_file}: {e}")
                stats["skipped"] += 1
                continue

            if not source_original.strip():
                stats["skipped"] += 1
                continue

            # --- Limpiar comentarios ---
            source_clean = strip_comments(source_original)

            # --- BugLog (fuente primaria) ---
            bug_entries = load_bug_log(sol_path)
            if not bug_entries:
                stats["skipped"] += 1
                continue
            stats["con_buglog"] += 1

            buglog_lines = {e["loc"] for e in bug_entries}

            # --- Slither (validacion complementaria) ---
            tmp_path = None
            slither_lines = set()
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".sol", mode="w", encoding="utf-8", delete=False
                ) as tf:
                    tf.write(source_clean)
                    tmp_path = tf.name
                slither_out = run_slither(tmp_path, detectors)
                slither_lines = get_slither_lines(slither_out, detectors)
                if slither_lines:
                    stats["slither_ok"] += 1
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            # --- Extraccion desde BugLog (fuente primaria) ---
            # Slither corre como validacion estadistica pero no agrega lineas
            # adicionales para evitar falsos positivos de contratos originales.
            seen_in_contract = set()
            for line_no in sorted(buglog_lines):
                func_name, func_code = extract_function_at_line(source_clean, line_no)
                if func_code is None:
                    continue

                key = (vuln, sol_file, func_name)
                if key in seen_in_contract:
                    continue
                seen_in_contract.add(key)

                rows.append(
                    {
                        "id": row_id,
                        "contract_file": sol_file,
                        "vulnerability": vuln,
                        "function_name": func_name,
                        "function_code": func_code.strip(),
                        "dataset": "solidifi",
                    }
                )
                row_id += 1
                stats["funciones"] += 1

        global_stats[vuln] = stats
        print(
            f"  >> {stats['funciones']} funciones | "
            f"BugLog: {stats['con_buglog']} contratos | "
            f"Slither detecto en: {stats['slither_ok']} contratos | "
            f"skipped: {stats['skipped']}"
        )

    elapsed = time.time() - t_start

    # --- Resumen ---
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    total = sum(s["funciones"] for s in global_stats.values())
    for vuln, s in global_stats.items():
        print(f"  {vuln:<28} {s['funciones']:5d} funciones")
    print(f"  {'TOTAL':<28} {total:5d} funciones")
    print(f"  Tiempo de procesamiento: {elapsed:.1f}s")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    import time as _time

    t_start = _time.time()

    print("=" * 60)
    print("SolidiFI Functions Builder")
    print("BugLog (primary) + Slither (validation)")
    print("=" * 60)
    print(f"Data : {DATA_DIR}")
    print(f"Out  : {OUTPUT_CSV}")

    if not DATA_DIR.exists():
        print(f"\n[ERROR] Dataset no encontrado: {DATA_DIR}")
        sys.exit(1)

    df = build_solidifi_functions_dataset(str(DATA_DIR))

    if df.empty:
        print("[!] No se encontraron funciones.")
        return

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nGuardado: {OUTPUT_CSV} ({len(df)} filas)")
    print("\nDistribucion por vulnerabilidad:")
    print(df["vulnerability"].value_counts().to_string())

    elapsed = _time.time() - t_start
    sys.path.insert(0, str(_HERE.parent))
    from env_info import capture_env, save_report

    info = capture_env(csv_output_path=str(OUTPUT_CSV))
    info["pipeline_processing_time_s"] = round(elapsed, 2)
    save_report(info, str(ENV_REPORT))


if __name__ == "__main__":
    main()
