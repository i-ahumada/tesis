"""
makeSplit.py
------------------------------------
Genera los datasets finales de entrenamiento y test combinando los tres
datasets individuales producidos por los makeX.py correspondientes.

Replicación de: Tang et al., Scientific Reports 2023
  doi: 10.1038/s41598-023-47219-0

Pipeline:
  TEST  : solidifi_functions.csv  →  test_functions.csv  (5,117 funciones)
          Sin hash dedup: SolidiFI inyecta el mismo template buggy en múltiples
          contratos; aplicar dedup eliminaría el 87% de los casos válidos.

  TRAIN : smartbugs_functions.csv + slither_functions.csv
          Cross-dataset hash dedup:
            1. SmartBugs Wild se mantiene completo (base de referencia).
            2. Se eliminan de Slither Audited las funciones cuyo hash SHA-256
               (código normalizado) ya esté presente en SmartBugs.
            3. Resultado: 29,908 + 22,292 = 52,200 funciones.

  NOTA  : El solapamiento SolidiFI → train (~2%) es informacional. No se
          filtra, replicando el enfoque del paper.

Uso:
  # Con rutas por defecto (relativas a este script):
  python makeSplit.py

  # Con rutas explícitas:
  python makeSplit.py \\
      --smartbugs ../smartbugs/smartbugs_functions.csv \\
      --slither   ../slither/slither_functions.csv \\
      --solidifi  ../solidifi/solidifi_functions.csv \\
      --out_dir   .

  # Generar también env_report del split:
  python makeSplit.py --env_report

Salida:
  model/train_functions.csv    — 52,200 funciones (SmartBugs + Slither dedup)
  model/test_functions.csv     — 5,117 funciones  (SolidiFI, sin dedup)
  model/env_report_split.json  — entorno de ejecución (con --env_report)
"""

import hashlib
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas por defecto (relativas a este script en preprocessing/model/)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

SMARTBUGS_CSV = _HERE / ".." / "smartbugs" / "smartbugs_functions.csv"
SLITHER_CSV = _HERE / ".." / "slither" / "slither_functions.csv"
SOLIDIFI_CSV = _HERE / ".." / "solidifi" / "solidifi_functions.csv"
OUT_DIR = _HERE
ENV_REPORT = _HERE / "env_report_split.json"

VULN_CLASSES = [
    "Re-entrancy",
    "Timestamp-Dependency",
    "Unhandled-Exceptions",
    "tx.origin",
]


# ---------------------------------------------------------------------------
# Hash de normalización (mismo criterio que makeSmartbugs/makeSlither)
# ---------------------------------------------------------------------------


def normalize_function_code(code: str) -> str:
    """
    Normaliza el formato del código fuente para consistencia entre datasets:
      - Tabs → 4 espacios
      - Elimina espacios al final de cada línea
      - Colapsa líneas consecutivas en blanco a una sola
      - Elimina líneas en blanco al inicio y final
    """
    lines = str(code).expandtabs(4).splitlines()
    lines = [line.rstrip() for line in lines]
    normalized = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        normalized.append(line)
        prev_blank = is_blank
    return "\n".join(normalized).strip()


def _normalize(code: str) -> str:
    """Colapsa whitespace para comparación semántica de código (dedup)."""
    return re.sub(r"\s+", " ", str(code)).strip()


def _hash(code: str) -> str:
    return hashlib.sha256(_normalize(code).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_datasets(
    smartbugs_csv: Path,
    slither_csv: Path,
    solidifi_csv: Path,
    out_dir: Path,
) -> tuple:
    """
    Retorna (train_df, test_df) y guarda los CSVs en out_dir.
    """
    t_total = time.time()

    # ------------------------------------------------------------------ carga
    print("\n[1/4] Cargando CSVs de entrada...")
    df_sb = pd.read_csv(smartbugs_csv, encoding="utf-8")
    df_sl = pd.read_csv(slither_csv, encoding="utf-8")
    df_sol = pd.read_csv(solidifi_csv, encoding="utf-8")

    print(f"  SmartBugs Wild   : {len(df_sb):>6,} funciones")
    print(f"  Slither Audited  : {len(df_sl):>6,} funciones")
    print(f"  SolidiFI         : {len(df_sol):>6,} funciones")
    print(f"  Total inputs     : {len(df_sb) + len(df_sl) + len(df_sol):>6,}")

    # Verificar columnas requeridas
    required = {"function_code", "vulnerability"}
    for name, df in [("SmartBugs", df_sb), ("Slither", df_sl), ("SolidiFI", df_sol)]:
        missing = required - set(df.columns)
        if missing:
            print(f"\n[ERROR] {name} le faltan columnas: {missing}")
            sys.exit(1)

    # Normalizar formato de código antes del dedup y guardado
    print("\n[1b/4] Normalizando formato de código...")
    for df in [df_sb, df_sl, df_sol]:
        df["function_code"] = df["function_code"].apply(normalize_function_code)

    # ----------------------------------------------------------------- test
    print("\n[2/4] Construyendo test set (SolidiFI, sin dedup)...")
    test_df = df_sol.copy().reset_index(drop=True)

    print(f"  Test : {len(test_df):,} funciones")
    print("  Distribución test:")
    vc = test_df["vulnerability"].value_counts()
    for vuln in VULN_CLASSES:
        n = vc.get(vuln, 0)
        print(f"    {vuln:<30}: {n:>5,}  ({100 * n / len(test_df):.1f}%)")

    # ----------------------------------------------------------------- train
    print("\n[3/4] Construyendo train set (SmartBugs + Slither Audited)...")

    # Hashes de SmartBugs — base de referencia, se mantiene completo
    print("  Computando hashes SmartBugs Wild...")
    t0 = time.time()
    df_sb["_hash"] = df_sb["function_code"].apply(_hash)
    sb_hashes = set(df_sb["_hash"])
    print(f"  {len(sb_hashes):,} hashes únicos ({time.time() - t0:.1f}s)")

    # Dedup cross-dataset: eliminar de Slither lo que ya está en SmartBugs
    print("  Computando hashes Slither Audited...")
    t0 = time.time()
    df_sl["_hash"] = df_sl["function_code"].apply(_hash)
    print(f"  {len(set(df_sl['_hash'])):,} hashes únicos Slither ({time.time() - t0:.1f}s)")

    n_before = len(df_sl)
    df_sl_dedup = df_sl[~df_sl["_hash"].isin(sb_hashes)].copy()
    n_removed = n_before - len(df_sl_dedup)

    print(f"\n  Slither Audited antes dedup : {n_before:,}")
    print(f"  Duplicados eliminados       : {n_removed:,}  ({100 * n_removed / n_before:.1f}%)")
    print(f"  Slither Audited final       : {len(df_sl_dedup):,}")

    # Solapamiento informacional SolidiFI → train (no se filtra)
    sol_hashes = set(df_sol["function_code"].apply(_hash))
    train_hashes = sb_hashes | set(df_sl_dedup["_hash"])
    overlap = len(sol_hashes & train_hashes)
    print(
        f"\n  Solapamiento test → train   : {overlap} funciones ({100 * overlap / len(sol_hashes):.1f}%) — informacional, no filtrado"
    )

    # Concatenar train
    train_df = pd.concat(
        [
            df_sb.drop(columns=["_hash"]),
            df_sl_dedup.drop(columns=["_hash"]),
        ],
        ignore_index=True,
    )

    print(f"\n  Train total: {len(train_df):,} funciones")
    print("  Distribución train:")
    vc_train = train_df["vulnerability"].value_counts()
    for vuln in VULN_CLASSES:
        n = vc_train.get(vuln, 0)
        pct = 100 * n / len(train_df)
        bar = "█" * int(pct / 2)
        print(f"    {vuln:<30}: {n:>6,}  ({pct:5.1f}%)  {bar}")

    # ----------------------------------------------------------------- guarda
    print("\n[4/4] Guardando...")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_functions.csv"
    test_path = out_dir / "test_functions.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    elapsed = time.time() - t_total
    print(f"  Guardado: {train_path}  ({len(train_df):,} filas)")
    print(f"  Guardado: {test_path}   ({len(test_df):,} filas)")
    print(f"  Tiempo total: {elapsed:.1f}s")

    return train_df, test_df, elapsed


# ---------------------------------------------------------------------------
# Env report (opcional)
# ---------------------------------------------------------------------------


def generate_env_report(train_path: Path, test_path: Path, processing_time_s: float):
    """Documenta el entorno del split en ENV_REPORT."""
    sys.path.insert(0, str(_HERE / ".."))
    from env_info import capture_env, save_report

    info = capture_env(csv_output_path=str(train_path))
    info["pipeline_processing_time_s"] = round(processing_time_s, 2)
    info["test_csv"] = {"path": str(test_path.resolve())}
    try:
        df_t = pd.read_csv(test_path)
        info["test_csv"]["rows"] = len(df_t)
        info["test_csv"]["vuln_counts"] = df_t["vulnerability"].value_counts().to_dict()
    except Exception:
        pass

    save_report(info, str(ENV_REPORT))
    print(f"  Env report: {ENV_REPORT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Dataset Split Builder")
    print("Train : SmartBugs Wild + Slither Audited (cross-dataset hash dedup)")
    print("Test  : SolidiFI (sin hash dedup)")
    print("=" * 60)
    print(f"SmartBugs : {SMARTBUGS_CSV}")
    print(f"Slither   : {SLITHER_CSV}")
    print(f"SolidiFI  : {SOLIDIFI_CSV}")
    print(f"Salida    : {OUT_DIR}")

    for path, name in [
        (SMARTBUGS_CSV, "SmartBugs"),
        (SLITHER_CSV, "Slither"),
        (SOLIDIFI_CSV, "SolidiFI"),
    ]:
        if not path.exists():
            print(f"\n[ERROR] No encontrado: {path}")
            print(f"  Ejecutá primero make{name}.py correspondiente.")
            sys.exit(1)

    train_df, test_df, elapsed = build_datasets(SMARTBUGS_CSV, SLITHER_CSV, SOLIDIFI_CSV, OUT_DIR)

    print("\nGenerando env_report...")
    generate_env_report(
        train_path=OUT_DIR / "train_functions.csv",
        test_path=OUT_DIR / "test_functions.csv",
        processing_time_s=elapsed,
    )

    print("\n" + "=" * 60)
    print("Completado.")
    print(f"  train_functions.csv : {len(train_df):,} funciones")
    print(f"  test_functions.csv  : {len(test_df):,} funciones")
    print("=" * 60)


if __name__ == "__main__":
    main()
