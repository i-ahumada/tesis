"""
makeSmartbugs.py
------------------------------------
Procesa el dataset SmartBugs-Wild y genera un CSV con funciones vulnerables
para las 4 vulnerabilidades objetivo, siguiendo la metodologia del paper:

  "Deep learning-based solution for smart contract vulnerabilities detection"
  (Tang et al., Scientific Reports 2023, doi:10.1038/s41598-023-47219-0)

El paper usa Slither para extraer ~30,000 funciones vulnerables del conjunto
de entrenamiento (SmartBugs-Wild + Slither Audited Dataset).

Metodologia:
  1. Cargar el CSV de SmartBugs Wild.
  2. Opcionalmente pre-filtrar contratos por etiquetas MCD (vulnerability_category_labels).
     Con --no_filter se omite este paso y se procesan todos los contratos.
  3. Para cada contrato:
     a. Si el source_code viene en una sola linea (< 5 newlines), reformatear
        insertando \n despues de { } ; para que Slither reporte lineas correctas.
        Si ya es multi-linea (dataset original), se usa directamente.
     b. Eliminar comentarios preservando los numeros de linea.
     c. Detectar la version de solc compatible con el pragma.
     d. Ejecutar Slither con los detectores objetivo.
     e. Para cada linea vulnerable reportada, extraer la funcion contenedora.
  4. Deduplicar por hash SHA-256 del codigo normalizado.
  5. Guardar CSV con columnas: id, contract_file, vulnerability,
     function_name, function_code, dataset

Datasets de entrada soportados:
  - smartbugs_wild.csv          (columnas: address, source_code, tools, ...)

Uso:
  # Dataset completo sin filtro MCD (recomendado para reproducir el paper):
  python makeSmartbugs.py --input datasets/smartbugs_wild.csv
                          --output func_dataset/smartbugs_functions.csv
                          --no_filter

  # Dataset pre-filtrado con etiquetas MCD:
  python makeSmartbugs.py --input datasets/discard_plus512_t12_mu03_train.csv
                          --output func_dataset/smartbugs_functions.csv
                          --txorigin_scan_all

Prerequisitos:
  pip install slither-analyzer solc-select pandas packaging
  solc-select install 0.4.25 0.4.26 0.5.17 0.6.12 0.7.6 0.8.17
"""

import ast
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

# Categorias del CSV que indican posible presencia de nuestras vulnerabilidades
# (usadas para pre-filtrar y reducir el numero de contratos a procesar)
CATEGORY_FILTER = {
    "Re-entrancy": {"reentrancy"},
    "Timestamp-Dependency": {"time_manipulation"},
    "Unhandled-Exceptions": {"unchecked_low_calls"},
    "tx.origin": {"other"},  # tx.origin se mapea a 'other' en SmartBugs
}

# Versiones de solc instaladas (en orden descendente, para preferir las mas nuevas)
INSTALLED_SOLC = ["0.8.17", "0.7.6", "0.6.12", "0.6.6", "0.5.17", "0.4.26", "0.4.25"]

# Directorio de binarios de solc instalados por solc-select
_SOLC_ARTIFACTS = os.path.join(os.path.expanduser("~"), ".solc-select", "artifacts")

# ---------------------------------------------------------------------------
# Rutas (sin argumentos — ajustar si se mueve el dataset)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
INPUT_CSV = _HERE / "data" / "smartbugs_wild.csv"
OUTPUT_CSV = _HERE / "smartbugs_functions.csv"
ENV_REPORT = _HERE / "env_report_smartbugs.json"

# ---------------------------------------------------------------------------
# Reformateador de Solidity (single-line -> multi-line)
# ---------------------------------------------------------------------------


def basic_sol_reformat(code: str) -> str:
    """
    Convierte Solidity de una sola linea a multi-linea insertando \n
    despues de '{', '}' y ';'. Respeta strings literales (no inserta dentro).
    """
    out = []
    in_str, str_char = False, ""
    for i, c in enumerate(code):
        if not in_str and c in ('"', "'"):
            in_str, str_char = True, c
            out.append(c)
        elif in_str and c == str_char and (i == 0 or code[i - 1] != "\\"):
            in_str = False
            out.append(c)
        else:
            out.append(c)
            if not in_str and c in ("{", "}", ";"):
                out.append("\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Limpieza de comentarios (preserva numeros de linea)
# ---------------------------------------------------------------------------


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
# Seleccion automatica de version de solc por pragma
# ---------------------------------------------------------------------------

_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);")


def _parse_constraints(pragma_str: str):
    """Convierte un pragma string a lista de (operador, version)."""
    constraints = []
    for part in pragma_str.strip().split():
        part = part.strip()
        if not part:
            continue
        if part.startswith("^"):
            ver = part[1:]
            segs = ver.split(".")
            try:
                if segs[0] == "0":
                    upper = f"0.{int(segs[1]) + 1}.0"
                else:
                    upper = f"{int(segs[0]) + 1}.0.0"
            except (IndexError, ValueError):
                upper = "99.0.0"
            constraints.append((">=", ver))
            constraints.append(("<", upper))
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


def _solc_bin(version: str) -> str:
    """Retorna el path al binario solc para la version dada."""
    return os.path.join(_SOLC_ARTIFACTS, f"solc-{version}", f"solc-{version}")


def detect_solc_version(source: str) -> str:
    """
    Retorna la mejor version instalada de solc para el pragma del contrato.
    Retorna el PATH AL BINARIO (requerido por --solc de Slither).
    """
    m = _PRAGMA_RE.search(source)
    fallback = _solc_bin("0.5.17")
    if not m:
        return fallback
    constraints = _parse_constraints(m.group(1))
    if not constraints:
        return fallback
    for v in INSTALLED_SOLC:
        try:
            pv = Version(v)
            ok = all(
                (op == ">=" and pv >= Version(bound))
                or (op == ">" and pv > Version(bound))
                or (op == "<=" and pv <= Version(bound))
                or (op == "<" and pv < Version(bound))
                or (op == "==" and pv == Version(bound))
                for op, bound in constraints
            )
            if ok:
                bin_path = _solc_bin(v)
                if os.path.exists(bin_path):
                    return bin_path
        except Exception:
            continue
    return fallback


# ---------------------------------------------------------------------------
# Ejecucion de Slither
# ---------------------------------------------------------------------------


def run_slither(sol_path: str, detectors: list, solc_bin: str) -> dict:
    """
    Corre Slither con el binario solc indicado.
    Retorna JSON parseado o {}.
    Nota: Slither devuelve codigo != 0 cuando detecta bugs (comportamiento normal).
    """
    cmd = [
        "slither",
        sol_path,
        "--detect",
        ",".join(detectors),
        "--solc",
        solc_bin,
        "--json",
        "-",
        "--exclude-dependencies",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        stdout = r.stdout.strip()
        if stdout:
            return json.loads(stdout)
    except subprocess.TimeoutExpired:
        pass
    except json.JSONDecodeError:
        pass
    except FileNotFoundError:
        raise RuntimeError("slither no encontrado. pip install slither-analyzer")
    return {}


def get_slither_lines(slither_output: dict, detectors: list, last_contract_name: str = None) -> set:
    """Extrae lineas reportadas por Slither para los detectores dados.

    Si last_contract_name se provee, solo incluye hallazgos cuya funcion
    pertenece a ese contrato (filtra librerias auxiliares del mismo archivo).
    """
    lines = set()
    det_set = set(detectors)
    for det in slither_output.get("results", {}).get("detectors", []):
        if det.get("check") not in det_set:
            continue
        elements = det.get("elements", [])
        if last_contract_name:
            func_el = next((e for e in elements if e.get("type") == "function"), None)
            if func_el is None:
                continue
            parent = func_el.get("type_specific_fields", {}).get("parent", {})
            if parent.get("name") != last_contract_name:
                continue
        for e in elements:
            for ln in e.get("source_mapping", {}).get("lines", []):
                lines.add(ln)
    return lines


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
    depth, opened = 0, False
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
    """Retorna (name, code) de la funcion mas cercana a target_line (1-indexed)."""
    target_0 = target_line - 1
    functions = _get_all_functions(source)
    if not functions:
        return None, None
    # Contenedora exacta
    for name, start, end, code in functions:
        if start <= target_0 <= end:
            return name, code
    # Mas cercana (hacia adelante o atras)
    best, best_dist = None, float("inf")
    for name, start, end, code in functions:
        dist = (start - target_0) if start > target_0 else (target_0 - end)
        if dist < best_dist:
            best_dist, best = dist, (name, code)
    return best if best else (None, None)


# ---------------------------------------------------------------------------
# Deduplicacion por hash
# ---------------------------------------------------------------------------


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())


def code_hash(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pre-filtrado de contratos
# ---------------------------------------------------------------------------


def _get_labels(label_str) -> set:
    """Parsea la columna vulnerability_category_labels a set de strings."""
    try:
        v = ast.literal_eval(str(label_str))
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return set(v)
        return set()
    except Exception:
        return set()


def _has_target_category(labels: set) -> bool:
    all_cats = set()
    for cats in CATEGORY_FILTER.values():
        all_cats |= cats
    return bool(labels & all_cats)


def _vulns_for_labels(labels: set) -> list:
    """Retorna lista de vulnerabilidades objetivo presentes en las labels."""
    result = []
    for vuln, cats in CATEGORY_FILTER.items():
        if labels & cats:
            result.append(vuln)
    return result


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def _process_contracts_for_vulns(
    df: pd.DataFrame,
    target_vulns_fn,  # callable(labels) -> list[str]
    stats: dict,
    seen_hashes: set,
    rows: list,
    row_id_start: int,
    n_no_source_ref: list,
    n_failed_ref: list,
) -> int:
    """
    Itera sobre df procesando cada contrato con Slither.
    Actualiza stats, seen_hashes y rows en lugar.
    Retorna el proximo row_id disponible.
    """
    row_id = row_id_start
    for idx, contract_row in df.iterrows():
        addr = str(contract_row.get("address", f"contract_{idx}"))
        src_raw = str(contract_row.get("source_code", ""))
        labels = contract_row.get("_labels", set())

        if not src_raw or src_raw.strip() in ("", "nan"):
            n_no_source_ref[0] += 1
            continue

        target_vulns = target_vulns_fn(labels)
        if not target_vulns:
            continue

        # Reformatear solo si el source viene en una sola linea (CSV filtrado).
        # El dataset original ya tiene multi-linea: aplicar reformat romperia
        # los numeros de linea que Slither reporta.
        if src_raw.count("\n") < 5:
            src_fmt = basic_sol_reformat(src_raw)
        else:
            src_fmt = src_raw
        src_clean = strip_comments(src_fmt)
        last_contract_name = get_last_contract_name(src_clean)
        solc_ver = detect_solc_version(src_raw)

        tmp_path = None
        slither_results_per_vuln = {}
        try:
            with tempfile.NamedTemporaryFile(suffix=".sol", mode="w", encoding="utf-8", delete=False) as tf:
                tf.write(src_clean)
                tmp_path = tf.name

            all_detectors = []
            for v in target_vulns:
                all_detectors.extend(SLITHER_DETECTORS[v])

            slither_out = run_slither(tmp_path, all_detectors, solc_ver)

            for v in target_vulns:
                lines = get_slither_lines(slither_out, SLITHER_DETECTORS[v], last_contract_name)
                slither_results_per_vuln[v] = lines

        except Exception:
            n_failed_ref[0] += 1
            continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        for vuln, vuln_lines in slither_results_per_vuln.items():
            if not vuln_lines:
                continue
            stats[vuln]["slither_hit"] += 1

            seen_in_contract = set()
            for line_no in sorted(vuln_lines):
                func_name, func_code = extract_function_at_line(src_clean, line_no)
                if func_code is None:
                    continue

                key_local = (vuln, addr, func_name)
                if key_local in seen_in_contract:
                    continue
                seen_in_contract.add(key_local)

                h = code_hash(func_code)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                rows.append(
                    {
                        "id": row_id,
                        "contract_file": addr,
                        "vulnerability": vuln,
                        "function_name": func_name,
                        "function_code": func_code.strip(),
                        "dataset": "smartbugs_wild",
                    }
                )
                row_id += 1
                stats[vuln]["funcs"] += 1

    return row_id


def build_smartbugs_functions_dataset(
    csv_path: str,
    max_contracts: int = None,
    txorigin_scan_all: bool = False,
    no_filter: bool = False,
) -> pd.DataFrame:
    """
    Lee el CSV de SmartBugs Wild y extrae funciones vulnerables usando Slither.

    no_filter=True: omite el pre-filtro MCD y ejecuta los 4 detectores sobre
      TODOS los contratos del CSV. Acepta tanto el formato original (smartbugs_wild.csv
      sin vulnerability_category_labels) como el formato pre-filtrado.

    txorigin_scan_all=True (solo cuando no_filter=False): segundo pase ejecutando
      el detector tx-origin sobre todos los contratos para maximizar cobertura.
    """
    print(f"Cargando: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"Contratos totales en CSV: {len(df_raw)}")

    if no_filter:
        # Sin filtro: procesar todos con los 4 detectores
        df_raw["_labels"] = [set()] * len(df_raw)
        df_target = df_raw.copy()
        print(f"Modo sin filtro: procesando todos los {len(df_target)} contratos")

        def _all_vulns(labels):
            return VULNERABILIDADES_OBJETIVO

        target_vulns_fn = _all_vulns
    else:
        # Pre-filtrar por categoria relevante (requiere vulnerability_category_labels)
        df_raw["_labels"] = df_raw["vulnerability_category_labels"].apply(_get_labels)
        df_target = df_raw[df_raw["_labels"].apply(_has_target_category)].copy()
        df_target = df_target.reset_index(drop=True)
        print(f"Contratos con categorias objetivo: {len(df_target)}")
        target_vulns_fn = _vulns_for_labels

    if max_contracts is not None:
        df_target = df_target.head(max_contracts)
        print(f"Limitado a {max_contracts} contratos para prueba")

    rows = []
    seen_hashes = set()
    row_id = 0
    t_start = time.time()

    stats = {v: {"slither_hit": 0, "funcs": 0} for v in VULNERABILIDADES_OBJETIVO}
    n_no_source = [0]
    n_failed = [0]

    # --- Pase principal ---
    row_id = _process_contracts_for_vulns(
        df=df_target,
        target_vulns_fn=target_vulns_fn,
        stats=stats,
        seen_hashes=seen_hashes,
        rows=rows,
        row_id_start=row_id,
        n_no_source_ref=n_no_source,
        n_failed_ref=n_failed,
    )

    # --- Pase suplementario tx.origin (solo en modo con filtro MCD) ---
    if txorigin_scan_all and not no_filter:
        print("\n" + "=" * 60)
        print("PASE SUPLEMENTARIO: tx-origin sobre todos los contratos")
        print("=" * 60)
        print(f"Escaneando {len(df_raw)} contratos con detector tx-origin...")

        txo_stats_before = stats["tx.origin"]["funcs"]

        def _only_txorigin(labels):
            return ["tx.origin"]

        row_id = _process_contracts_for_vulns(
            df=df_raw,
            target_vulns_fn=_only_txorigin,
            stats=stats,
            seen_hashes=seen_hashes,
            rows=rows,
            row_id_start=row_id,
            n_no_source_ref=n_no_source,
            n_failed_ref=n_failed,
        )

        txo_added = stats["tx.origin"]["funcs"] - txo_stats_before
        print(f"  Funciones tx.origin nuevas encontradas: {txo_added}")

    # Reasignar ids correlativos
    for i, r in enumerate(rows):
        r["id"] = i

    elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for v, s in stats.items():
        print(f"  {v:<28} slither_hit={s['slither_hit']:4d}  funcs={s['funcs']:5d}")
    total = sum(s["funcs"] for s in stats.values())
    print(f"  {'TOTAL':<28} {'':12s}  funcs={total:5d}")
    print(f"  Contratos sin source_code: {n_no_source[0]}")
    print(f"  Contratos fallidos (Slither): {n_failed[0]}")
    print(f"  Tiempo: {elapsed:.1f}s")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()

    print("=" * 60)
    print("SmartBugs Wild Functions Builder - Slither")
    print("Modo: sin filtro MCD (4 detectores sobre todos los contratos)")
    print("=" * 60)
    print(f"Input : {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")

    if not INPUT_CSV.exists():
        print(f"\n[ERROR] Dataset no encontrado: {INPUT_CSV}")
        sys.exit(1)

    df = build_smartbugs_functions_dataset(
        str(INPUT_CSV),
        max_contracts=None,
        txorigin_scan_all=False,
        no_filter=True,
    )

    if df.empty:
        print("[!] No se encontraron funciones vulnerables.")
        return

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nGuardado: {OUTPUT_CSV} ({len(df)} filas)")
    print("\nDistribucion por vulnerabilidad:")
    print(df["vulnerability"].value_counts().to_string())

    elapsed = time.time() - t_start
    sys.path.insert(0, str(_HERE.parent))
    from env_info import capture_env, save_report

    info = capture_env(csv_output_path=str(OUTPUT_CSV))
    info["pipeline_processing_time_s"] = round(elapsed, 2)
    save_report(info, str(ENV_REPORT))


if __name__ == "__main__":
    main()
