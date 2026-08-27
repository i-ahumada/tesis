"""
makeSlither.py — Extrae funciones vulnerables del dataset Slither-Audited
(mwritescode/slither-audited-smart-contracts, parquets en data/).

Decisiones de diseño:
- Se procesan TODOS los contratos con Slither, sin prefiltro por los resultados
  precomputados del dataset (ese prefiltro omitía el detector timestamp).
- Paralelismo con ThreadPoolExecutor y un worker por CPU: el trabajo pesado corre
  en el subproceso slither/solc, por lo que el GIL no es cuello de botella.
- Detectores según la Tabla 1 del paper (Tang et al. 2023), sin filtro por confianza.
- Normalización ANTES de ejecutar Slither: comentarios eliminados y código
  re-indentado con jsbeautifier. Así las líneas que reporta Slither corresponden
  al mismo texto del que se extraen las funciones.
- Filtro de contrato principal: se conservan los hallazgos cuya función pertenece
  al contrato principal o a su cadena de herencia — el código escrito por el
  programador. El principal se identifica como el `contract` con la clausura de
  herencia más grande (más robusto que "el último declarado", que a veces es un
  helper). Se descartan libraries y contratos auxiliares no heredados (SafeMath,
  interfaces sueltas, etc.) pegados en el mismo source_code.
- Cada fila guarda tanto la categoria de vulnerabilidad (clave de
  DETECTORS_BY_VULN, ej. "Unhandled-Exceptions") como el detector puntual de
  Slither que la genero (columna `detector`, ej. "unchecked-lowlevel"), para
  poder distinguir entre los detectores agrupados bajo una misma categoria.
- Dedup: (contrato, función, vulnerabilidad) y hash SHA-256 global del código.
  El detector no forma parte de la clave: si dos detectores de la misma
  categoria coinciden en la misma función, se conserva solo el primero.
- Corrida única, sin checkpoint (decisión acordada).

Uso: python makeSlither.py
Requisitos: pip install slither-analyzer pandas pyarrow jsbeautifier packaging
            solc-select install <versiones>
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jsbeautifier
import pandas as pd
from packaging.version import Version

DETECTORS_BY_VULN = {
    "Re-entrancy": ["reentrancy-eth", "reentrancy-no-eth", "reentrancy-unlimited-gas", "reentrancy-benign"],
    "Timestamp-Dependency": ["timestamp"],
    "Unhandled-Exceptions": ["unchecked-lowlevel", "unchecked-send"],
    "tx.origin": ["tx-origin"],
}
DETECTOR_TO_VULN = {det: vuln for vuln, dets in DETECTORS_BY_VULN.items() for det in dets}
ALL_DETECTORS = ",".join(DETECTOR_TO_VULN)

_HERE = Path(__file__).parent
PARQUET_DIR = _HERE / "data"
OUTPUT_CSV = _HERE / "slither_functions.csv"
DATASET_NAME = "slither_audited"

WORKERS = os.cpu_count() or 4
SLITHER_TIMEOUT = 120

_BEAUTIFY_OPTS = jsbeautifier.default_options()
_BEAUTIFY_OPTS.indent_size = 4

# ---------------------------------------------------------------------------
# Seleccion de solc por pragma
# ---------------------------------------------------------------------------

_SOLC_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"
_INSTALLED_SOLC = sorted(
    (p.name.removeprefix("solc-") for p in _SOLC_ARTIFACTS.glob("solc-*")),
    key=Version,
    reverse=True,
)
_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);")


def _parse_constraints(pragma_str: str) -> list:
    """Convierte el pragma en lista de (operador, version)."""
    constraints = []
    for part in pragma_str.split():
        if part.startswith("^"):
            ver = part[1:]
            segs = ver.split(".")
            try:
                upper = f"0.{int(segs[1]) + 1}.0" if segs[0] == "0" else f"{int(segs[0]) + 1}.0.0"
            except (IndexError, ValueError):
                upper = "99.0.0"
            constraints += [(">=", ver), ("<", upper)]
        elif part[:2] in (">=", "<="):
            constraints.append((part[:2], part[2:]))
        elif part[0] in (">", "<"):
            constraints.append((part[0], part[1:]))
        elif part.startswith("="):
            constraints.append(("==", part[1:]))
        elif re.match(r"^\d", part):
            constraints.append(("==", part))
    return constraints


_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def detect_solc_bin(source: str) -> str:
    """Path al binario solc instalado mas nuevo que satisface el pragma."""
    fallback = str(_SOLC_ARTIFACTS / "solc-0.5.17" / "solc-0.5.17")
    m = _PRAGMA_RE.search(source)
    if not m:
        return fallback
    constraints = _parse_constraints(m.group(1))
    if not constraints:
        return fallback
    for v in _INSTALLED_SOLC:
        try:
            if all(_OPS[op](Version(v), Version(bound)) for op, bound in constraints):
                return str(_SOLC_ARTIFACTS / f"solc-{v}" / f"solc-{v}")
        except Exception:
            continue
    return fallback


# ---------------------------------------------------------------------------
# Normalizacion y Slither
# ---------------------------------------------------------------------------


def strip_comments(source: str) -> str:
    """Elimina // y /* */ preservando los saltos de linea."""
    source = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def normalize_source(source: str) -> str:
    return strip_comments(jsbeautifier.beautify(source, _BEAUTIFY_OPTS))


_DECL_RE = re.compile(r"\b(contract|interface|library)\s+(\w+)(\s+is\s+[^{]+)?")


def main_contract_names(src_code: str) -> set:
    """Nombres del contrato principal y de toda su cadena de herencia.

    El principal es el `contract` con la clausura de herencia mas grande
    (empate: el declarado mas al final). Ese criterio es mas robusto que tomar
    el ultimo declarado, que a veces es un helper y no el contrato principal.
    Quedan fuera las libraries y los contratos no heredados.
    Set vacio = no filtrar (no se encontro ningun contract).
    """
    decls = _DECL_RE.findall(src_code)
    bases = {name: (re.findall(r"\w+", is_clause)[1:] if is_clause else []) for _, name, is_clause in decls}

    def closure(root: str) -> set:
        keep, stack = set(), [root]
        while stack:
            name = stack.pop()
            if name in keep or name not in bases:
                continue
            keep.add(name)
            stack.extend(bases[name])
        return keep

    contracts = [name for kind, name, _ in decls if kind == "contract"]
    if not contracts:
        return set()
    return max((closure(c) for c in reversed(contracts)), key=len)


def run_slither(src_normalized: str, solc_bin: str, keep_contracts: set) -> list:
    """
    Ejecuta Slither sobre el texto normalizado y retorna
    [(detector, nombre_funcion, linea_inicio, linea_fin)]. Vacio si falla.

    Si keep_contracts no esta vacio, descarta hallazgos cuya funcion no
    pertenece a esos contratos (filtra libraries y contratos auxiliares).
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".sol", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(src_normalized)
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SLITHER_TIMEOUT)
        if not result.stdout.strip():
            return []
        findings = []
        for det in json.loads(result.stdout).get("results", {}).get("detectors", []):
            check = det.get("check")
            if check not in DETECTOR_TO_VULN:
                continue
            func_el = next((e for e in det.get("elements", []) if e.get("type") == "function"), None)
            if not func_el:
                continue
            if keep_contracts:
                parent = func_el.get("type_specific_fields", {}).get("parent", {})
                if parent.get("name") not in keep_contracts:
                    continue
            lines = func_el.get("source_mapping", {}).get("lines", [])
            if not lines:
                continue
            findings.append((check, func_el.get("name", ""), min(lines), max(lines)))
        return findings
    except Exception:
        return []
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def process_contract(address: str, source: str) -> list:
    """Normaliza, corre Slither y retorna las filas (sin dedup global)."""
    src_norm = normalize_source(source)
    src_lines = src_norm.splitlines()
    keep_contracts = main_contract_names(src_norm)
    rows = []
    for check, func_name, start, end in run_slither(src_norm, detect_solc_bin(source), keep_contracts):
        code = "\n".join(ln for ln in src_lines[max(0, start - 1) : min(len(src_lines), end)] if ln.strip())
        if code:
            rows.append(
                {
                    "contract_file": address,
                    "vulnerability": DETECTOR_TO_VULN[check],
                    "detector": check,
                    "function_name": func_name,
                    "function_code": code,
                    "dataset": DATASET_NAME,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def dedup(results: list) -> pd.DataFrame:
    """Dedup por (contrato, funcion, vuln) y por hash global del codigo."""
    seen_local, seen_hashes, rows = set(), set(), []
    for contract_rows in results:
        for r in contract_rows or []:
            key = (r["contract_file"], r["function_name"], r["vulnerability"])
            h = hashlib.sha256(re.sub(r"\s+", " ", r["function_code"]).encode()).hexdigest()
            if key in seen_local or h in seen_hashes:
                continue
            seen_local.add(key)
            seen_hashes.add(h)
            rows.append({"id": len(rows), **r})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    parquets = sorted(PARQUET_DIR.glob("*.parquet"))
    df_raw = pd.concat([pd.read_parquet(f, columns=["contracts", "source_code"]) for f in parquets], ignore_index=True)

    tasks = [
        (str(row.contracts), str(row.source_code))
        for row in df_raw.itertuples(index=False)
        if str(row.source_code).strip() not in ("", "nan", "None")
    ]
    print(f"{len(df_raw):,} contratos cargados, {len(tasks):,} con codigo | {WORKERS} workers")

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_contract, addr, src): i for i, (addr, src) in enumerate(tasks)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done:,}/{len(tasks):,} ({100 * done // len(tasks)}%)  {time.time() - t0:,.0f}s", end="\r")
    print()

    df = dedup(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nGuardado: {OUTPUT_CSV} ({len(df):,} filas, {time.time() - t0:,.0f}s)")
    print(df["vulnerability"].value_counts().to_string())


if __name__ == "__main__":
    main()
