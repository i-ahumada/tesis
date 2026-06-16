# -*- coding: utf-8 -*-
"""
makeSlither.py
------------------------------------
Extrae funciones vulnerables del dataset "Slither Audited Smart Contracts"
(mwritescode/slither-audited-smart-contracts) para las 4 vulnerabilidades
objetivo, replicando la metodologia del paper:

  "Deep learning-based solution for smart contract vulnerabilities detection"
  (Tang et al., Scientific Reports 2023, doi:10.1038/s41598-023-47219-0)

Pipeline:
  1. CARGA    : Lee los Parquet de data/raw/ (120,608 contratos).
                Columnas relevantes: contracts (address), source_code, results.

  2. PREFILTRO: Usa la columna `results` (analisis Slither pre-computado en el
                dataset, sin source_mapping) para descartar contratos donde
                ninguno de nuestros detectores objetivo reporto un hallazgo.
                Reduce ~52%% los contratos a analizar (57,884 candidatos).

                Nota: el detector `timestamp` NO esta en los resultados pre-
                computados. Los contratos con SOLO Timestamp-Dependency no seran
                incluidos. Cobertura de esa categoria proviene de SmartBugs Wild.

  3. SLITHER  : Para cada candidato:
                a. Patch minimo de sintaxis para compatibilidad con solc 0.8.17
                   (pragma -> ^0.8.0, .call.value, now, throw, suicide, etc.).
                   Ninguna transformacion agrega ni elimina lineas.
                b. Escribe source parcheado en .sol temporal.
                c. Ejecuta Slither con detectores objetivo y --solc solc-0.8.17.
                d. Parsea el JSON de salida completo (con source_mapping).

  4. EXTRAE   : Por cada hallazgo:
                - Busca el element de tipo "function" para obtener rango de lineas.
                - Extrae el codigo del source ORIGINAL (no el parcheado).
                - Dedup intra-contrato: (address, funcion, vuln).
                - Dedup global: SHA-256 del codigo normalizado.

  5. GUARDA   : CSV con columnas estandar:
                id, contract_file, vulnerability, function_name,
                function_code, dataset='slither_audited'

Detectores usados:
  Re-entrancy        -> reentrancy-eth, reentrancy-no-eth
  Timestamp-Dep.     -> timestamp  (cobertura parcial, ver nota en paso 2)
  Unhandled-Except.  -> unchecked-lowlevel, unchecked-send, unused-return
  tx.origin          -> tx-origin

Descarga del dataset:
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('mwritescode/slither-audited-smart-contracts',
      repo_type='dataset', local_dir='datasets/slither-audited',
      allow_patterns=['data/raw/*.parquet','data/label_mappings.json'])
  "

Uso:
  python makeSlither.py --parquet_dir datasets/slither-audited
                        --output func_dataset/slither_functions.csv

  # Prueba rapida con N contratos:
  python makeSlither.py --parquet_dir datasets/slither-audited
                        --output func_dataset/slither_functions.csv
                        --max_contracts 100

Prerequisitos:
  pip install slither-analyzer pandas pyarrow
  solc-select install 0.8.17
"""

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from packaging.version import Version

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

DETECTORS_BY_VULN = {
    "Re-entrancy": ["reentrancy-eth", "reentrancy-no-eth"],
    "Timestamp-Dependency": ["timestamp"],
    "Unhandled-Exceptions": ["unchecked-lowlevel", "unchecked-send", "unused-return"],
    "tx.origin": ["tx-origin"],
}

DETECTOR_TO_VULN = {det: vuln for vuln, dets in DETECTORS_BY_VULN.items() for det in dets}

ALL_DETECTORS = ",".join(d for dets in DETECTORS_BY_VULN.values() for d in dets)

# ---------------------------------------------------------------------------
# Rutas (sin argumentos — ajustar si se mueve el dataset)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
PARQUET_DIR = _HERE / "data"
OUTPUT_CSV = _HERE / "slither_functions.csv"
ENV_REPORT = _HERE / "env_report_slither.json"

# Detectores presentes en los resultados pre-computados del dataset.
# 'timestamp' tiene 0 ocurrencias en todo el dataset -> no se usa para prefiltro.
_PRECOMPUTED_DETECTORS = {
    "reentrancy-eth",
    "reentrancy-no-eth",
    "unchecked-lowlevel",
    "unchecked-send",
    "unused-return",
    "tx-origin",
}

# ---------------------------------------------------------------------------
# Deteccion de version solc y limpieza de comentarios
# (duplicado de makeSmartbugs.py para que el script sea autocontenido)
# ---------------------------------------------------------------------------

_SOLC_ARTIFACTS = os.path.join(os.path.expanduser("~"), ".solc-select", "artifacts")
_INSTALLED_SOLC = [
    "0.8.17",
    "0.8.13",
    "0.8.12",
    "0.8.11",
    "0.8.10",
    "0.8.9",
    "0.8.7",
    "0.8.6",
    "0.8.4",
    "0.8.3",
    "0.8.2",
    "0.8.0",
    "0.7.6",
    "0.7.5",
    "0.7.4",
    "0.7.0",
    "0.6.12",
    "0.6.11",
    "0.6.7",
    "0.6.6",
    "0.6.2",
    "0.6.0",
    "0.5.17",
    "0.5.16",
    "0.5.0",
    "0.4.26",
    "0.4.25",
    "0.4.24",
    "0.4.23",
    "0.4.21",
    "0.4.19",
    "0.4.18",
]
_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);")


def _solc_bin(version: str) -> str:
    return os.path.join(_SOLC_ARTIFACTS, f"solc-{version}", f"solc-{version}")


def _parse_constraints(pragma_str: str):
    constraints = []
    for part in pragma_str.strip().split():
        part = part.strip()
        if not part:
            continue
        if part.startswith("^"):
            ver = part[1:]
            segs = ver.split(".")
            try:
                upper = f"0.{int(segs[1]) + 1}.0" if segs[0] == "0" else f"{int(segs[0]) + 1}.0.0"
            except (IndexError, ValueError):
                upper = "99.0.0"
            constraints += [(">=", ver), ("<", upper)]
        elif part.startswith(">="):
            constraints.append((">=", part[2:]))
        elif part.startswith(">"):
            constraints.append((">", part[1:]))
        elif part.startswith("<="):
            constraints.append(("<=", part[2:]))
        elif part.startswith("<"):
            constraints.append(("<", part[1:]))
        elif part.startswith("="):
            constraints.append(("==", part[1:]))
        elif re.match(r"^\d", part):
            constraints.append(("==", part))
    return constraints


def detect_solc_version(source: str) -> str:
    """Retorna el path al binario solc mas adecuado para el pragma del contrato."""
    m = _PRAGMA_RE.search(source)
    fallback = _solc_bin("0.5.17")
    if not m:
        return fallback
    constraints = _parse_constraints(m.group(1))
    if not constraints:
        return fallback
    for v in _INSTALLED_SOLC:
        try:
            pv = Version(v)
            ok = all(
                (op == ">=" and pv >= Version(b))
                or (op == ">" and pv > Version(b))
                or (op == "<=" and pv <= Version(b))
                or (op == "<" and pv < Version(b))
                or (op == "==" and pv == Version(b))
                for op, b in constraints
            )
            if ok:
                bin_path = _solc_bin(v)
                if os.path.exists(bin_path):
                    return bin_path
        except Exception:
            continue
    return fallback


def strip_comments(source: str) -> str:
    """Elimina // y /* */ preservando los saltos de linea."""

    def replace_block(m):
        return "\n" * m.group(0).count("\n")

    source = re.sub(r"/\*.*?\*/", replace_block, source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    return source


def get_last_contract_name(src_code: str) -> str | None:
    """Retorna el nombre del ultimo contract/library declarado en el source."""
    matches = re.findall(r"\b(?:library|contract)\s+(\w+)", src_code)
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------


def load_parquets(parquet_dir: str) -> pd.DataFrame:
    """
    Carga todos los .parquet bajo parquet_dir (recursivo) y los concatena.
    """
    files = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron .parquet en: {parquet_dir}\n"
            "Descargalos con snapshot_download (ver docstring del modulo)."
        )
    print(f"  {len(files)} archivo(s) Parquet:")
    frames = []
    for f in files:
        df_part = pd.read_parquet(f)
        print(f"    {os.path.basename(f):35s} {len(df_part):>7,} contratos")
        frames.append(df_part)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Prefiltro por resultados pre-computados
# ---------------------------------------------------------------------------


def _precomputed_detectors(results_str) -> set:
    """
    Parsea la columna `results` (JSON string) y retorna el conjunto de
    detectores que reportaron hallazgos.
    """
    try:
        data = json.loads(str(results_str))
        return {d["check"] for d in data.get("results", {}).get("detectors", [])}
    except Exception:
        return set()


def is_candidate(results_str) -> bool:
    """
    True si el analisis pre-computado contiene al menos uno de nuestros
    detectores objetivo (excepto timestamp, que no esta en el dataset).
    """
    return bool(_precomputed_detectors(results_str) & _PRECOMPUTED_DETECTORS)


# ---------------------------------------------------------------------------
# Ejecucion de Slither
# ---------------------------------------------------------------------------


def run_slither(source: str, last_contract_name: str = None) -> list[dict]:
    """
    Escribe el source completo (con comentarios eliminados) en un .sol temporal,
    ejecuta Slither y retorna lista de findings del ultimo contrato:
        [{check, function_name, func_start_line, func_end_line, bug_line}, ...]

    Si last_contract_name se provee, solo incluye hallazgos cuya funcion
    pertenece a ese contrato. Retorna lista vacia si no hay hallazgos o Slither falla.
    """
    src_clean = strip_comments(source)
    solc_bin = detect_solc_version(source)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".sol", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(src_clean)
            tmp_path = tmp.name

        cmd = [
            "slither",
            tmp_path,
            "--json",
            "-",
            "--detect",
            ALL_DETECTORS,
            "--solc",
            solc_bin,
            "--exclude-dependencies",
            "--solc-disable-warnings",
            "--disable-color",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if not result.stdout.strip():
            return []

        data = json.loads(result.stdout)
        detectors = data.get("results", {}).get("detectors", [])
        if not detectors:
            return []

        findings = []
        for det in detectors:
            check = det.get("check", "")
            if check not in DETECTOR_TO_VULN:
                continue

            elements = det.get("elements", [])

            func_el = next((e for e in elements if e.get("type") == "function"), None)
            if not func_el:
                continue

            if last_contract_name:
                parent = func_el.get("type_specific_fields", {}).get("parent", {})
                if parent.get("name") != last_contract_name:
                    continue

            func_lines = func_el.get("source_mapping", {}).get("lines", [])
            if not func_lines:
                continue

            node_el = next((e for e in elements if e.get("type") == "node"), None)
            bug_line = None
            if node_el:
                node_lines = node_el.get("source_mapping", {}).get("lines", [])
                if node_lines:
                    bug_line = min(node_lines)

            findings.append(
                {
                    "check": check,
                    "function_name": func_el.get("name", ""),
                    "func_start_line": min(func_lines),
                    "func_end_line": max(func_lines),
                    "bug_line": bug_line,
                }
            )

        return findings

    except subprocess.TimeoutExpired:
        return []
    except (json.JSONDecodeError, Exception):
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Extraccion de codigo de funcion
# ---------------------------------------------------------------------------


def extract_function_code(source: str, start_line: int, end_line: int) -> str:
    """Extrae las lineas [start_line, end_line] del source original (1-indexed)."""
    lines = source.splitlines()
    start = max(0, start_line - 1)
    end = min(len(lines), end_line)
    return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_slither_functions_dataset(
    df_raw: pd.DataFrame,
    max_contracts: int = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Pipeline: prefiltro -> Slither -> extraccion -> dedup -> DataFrame.
    """
    # Validar columnas
    for col in ("source_code", "results"):
        if col not in df_raw.columns:
            raise ValueError(f"Columna requerida ausente: '{col}'")

    has_address = "contracts" in df_raw.columns

    # Prefiltro
    print("\n[2/4] Prefiltro por resultados pre-computados...")
    mask = df_raw["results"].apply(is_candidate)
    df_candidates = df_raw[mask].reset_index(drop=True)
    n_total = len(df_raw)
    n_candidates = len(df_candidates)
    print(f"  Total contratos      : {n_total:>8,}")
    print(f"  Candidatos           : {n_candidates:>8,}  ({100 * n_candidates / max(n_total, 1):.1f}%)")

    if max_contracts:
        df_candidates = df_candidates.head(max_contracts)
        print(f"  Limitado a           : {len(df_candidates):>8,}  (--max_contracts)")

    # Slither + extraccion
    print("\n[3/4] Ejecutando Slither y extrayendo funciones...")

    rows = []
    seen_local: set[tuple] = set()
    seen_hashes: set[str] = set()
    stats = {
        "sin_source": 0,
        "sin_hallazgos": 0,
        "con_hallazgos": 0,
        "por_vuln": {v: 0 for v in DETECTORS_BY_VULN},
    }
    row_id = 0
    n = len(df_candidates)

    for i, (_, row) in enumerate(df_candidates.iterrows(), 1):
        source = str(row.get("source_code") or "")
        address = str(row.get("contracts", f"row_{i}")) if has_address else f"row_{i}"

        if verbose:
            print(f"  [{i:6d}/{n}] {address[:40]}")
        elif i % 100 == 0 or i == n:
            print(f"  Procesados: {i:,}/{n:,} | hallazgos: {stats['con_hallazgos']:,}", end="\r")

        if not source.strip() or source.strip() in ("nan", "None"):
            stats["sin_source"] += 1
            continue

        last_contract_name = get_last_contract_name(source)
        findings = run_slither(source, last_contract_name=last_contract_name)

        if not findings:
            stats["sin_hallazgos"] += 1
            continue

        stats["con_hallazgos"] += 1

        for f in findings:
            check = f["check"]
            vuln = DETECTOR_TO_VULN[check]

            func_code = extract_function_code(source, f["func_start_line"], f["func_end_line"])
            if not func_code.strip():
                continue

            # Dedup intra-contrato
            key_local = (address, f["function_name"], vuln)
            if key_local in seen_local:
                continue
            seen_local.add(key_local)

            # Dedup global por hash
            normalized = re.sub(r"\s+", " ", func_code.strip())
            h = hashlib.sha256(normalized.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            rows.append(
                {
                    "id": row_id,
                    "contract_file": address,
                    "vulnerability": vuln,
                    "function_name": f["function_name"],
                    "function_code": func_code.strip(),
                    "dataset": "slither_audited",
                }
            )
            row_id += 1
            stats["por_vuln"][vuln] += 1

    print()

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Sin codigo fuente        : {stats['sin_source']:>6,}")
    print(f"  Sin hallazgos (Slither)  : {stats['sin_hallazgos']:>6,}")
    print(f"  Con >= 1 hallazgo        : {stats['con_hallazgos']:>6,}")
    print("\n  Funciones por vulnerabilidad:")
    for vuln, count in stats["por_vuln"].items():
        print(f"    {vuln:<30}: {count:>6,}")
    total = sum(stats["por_vuln"].values())
    print(f"    {'TOTAL':<30}: {total:>6,}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()

    print("=" * 60)
    print("Slither Audited Smart Contracts - Functions Builder")
    print("=" * 60)
    print(f"Parquet dir   : {PARQUET_DIR}")
    print(f"Output        : {OUTPUT_CSV}")
    print(f"Detectores    : {ALL_DETECTORS}")

    if not PARQUET_DIR.exists():
        print(f"\n[ERROR] Dataset no encontrado: {PARQUET_DIR}")
        sys.exit(1)

    print("\n[1/4] Cargando Parquet...")
    df_raw = load_parquets(str(PARQUET_DIR))
    print(f"  {len(df_raw):,} contratos cargados | columnas: {list(df_raw.columns)}")

    df_result = build_slither_functions_dataset(df_raw)

    print("\n[4/4] Guardando resultado...")
    if df_result.empty:
        print("  No se encontraron funciones vulnerables.")
        return

    df_result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"  Guardado: {OUTPUT_CSV} ({len(df_result):,} filas)")
    print("\nDistribucion por vulnerabilidad:")
    print(df_result["vulnerability"].value_counts().to_string())

    elapsed = time.time() - t_start
    sys.path.insert(0, str(_HERE.parent))
    from env_info import capture_env, save_report

    info = capture_env(csv_output_path=str(OUTPUT_CSV))
    info["pipeline_processing_time_s"] = round(elapsed, 2)
    save_report(info, str(ENV_REPORT))


if __name__ == "__main__":
    main()
