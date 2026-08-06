"""
makeSplit.py — Genera train_functions.csv y test_functions.csv finales.

Decisiones de diseño:
- Test: SolidiFI completo, sin dedup interno (el mismo template inyectado en
  varios contratos representa casos de test válidos).
- Train: SmartBugs-Wild + Slither-Audited, con dos deduplicaciones por hash
  SHA-256 del código con espacios colapsados:
    1. Entre datasets: se conserva SmartBugs y se eliminan de Slither las
       funciones cuyo hash ya está en SmartBugs.
    2. Test → Train: se elimina de train toda función cuyo hash aparece en
       test, para evitar data leakage.

Validaciones post-split (run_validations, aplicadas a train y test tras
generarse los datasets). Cada una es una función independiente que elimina
las filas problemáticas; para desactivar una, comentar su línea en
run_validations:
    1. check_starts_with_function: el código debe iniciar con function,
       constructor, fallback, receive o modifier.
    2. check_balanced_braces: nº de '{' debe igualar al de '}'.
    3. check_txorigin_label: elimina snippets con 'tx.origin' cuya etiqueta
       no es tx.origin.
    4. check_timestamp_label: elimina snippets con uso de timestamp
       (block.timestamp / now, ver _TIMESTAMP_RE) cuya etiqueta no es
       Timestamp-Dependency.

Uso: python makeSplit.py  (requiere haber corrido antes los tres makeX.py)
"""

import hashlib
import re
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
SMARTBUGS_CSV = _HERE.parent / "smartbugs" / "smartbugs_functions.csv"
SLITHER_CSV = _HERE.parent / "slither" / "slither_functions.csv"
SOLIDIFI_CSV = _HERE.parent / "solidifi" / "solidifi_functions.csv"
TRAIN_CSV = _HERE / "train_functions.csv"
TEST_CSV = _HERE / "test_functions.csv"


def code_hash(code: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", str(code)).strip().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validaciones post-split
#
# Cada validación elimina las filas problemáticas y devuelve el DataFrame
# filtrado. Se ejecutan sobre train y test tras generarse (run_validations).
# Para desactivar una validación, comentar su línea dentro de run_validations.
# ---------------------------------------------------------------------------

# Usos que dispara el detector 'timestamp' de Slither. Ajustar aquí si se
# quiere cambiar el criterio de la validación check_timestamp_label.
_TIMESTAMP_RE = re.compile(r"\bblock\.timestamp\b|\bnow\b")


# Prefijos válidos para un snippet: los cuatro tipos de función de Solidity
# (function, constructor, fallback, receive) más modifier, que no es una función
# pero cuyo cuerpo puede contener las vulnerabilidades objetivo.
_FUNCTION_PREFIXES = ("function", "constructor", "fallback", "receive", "modifier")


def check_starts_with_function(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Elimina snippets cuyo código no inicia con function/constructor/fallback/
    receive/modifier."""
    mask_ok = df["function_code"].apply(lambda c: str(c).lstrip().startswith(_FUNCTION_PREFIXES))
    removed = int((~mask_ok).sum())
    if removed:
        print(f"[{name}] formato inválido (no inicia con function/constructor/fallback/receive/modifier): -{removed:,}")
    return df[mask_ok].reset_index(drop=True)


def check_balanced_braces(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Elimina snippets con llaves desbalanceadas ('{' != '}')."""
    mask_ok = df["function_code"].apply(lambda c: str(c).count("{") == str(c).count("}"))
    removed = int((~mask_ok).sum())
    if removed:
        print(f"[{name}] llaves desbalanceadas: -{removed:,}")
    return df[mask_ok].reset_index(drop=True)


def check_txorigin_label(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Elimina snippets con 'tx.origin' cuya etiqueta no es tx.origin."""
    has_txo = df["function_code"].apply(lambda c: "tx.origin" in str(c))
    mask_bad = has_txo & (df["vulnerability"] != "tx.origin")
    removed = int(mask_bad.sum())
    if removed:
        print(f"[{name}] 'tx.origin' en etiqueta distinta a tx.origin: -{removed:,}")
    return df[~mask_bad].reset_index(drop=True)


def check_timestamp_label(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Elimina snippets con uso de timestamp (block.timestamp / now) cuya
    etiqueta no es Timestamp-Dependency."""
    has_ts = df["function_code"].apply(lambda c: bool(_TIMESTAMP_RE.search(str(c))))
    mask_bad = has_ts & (df["vulnerability"] != "Timestamp-Dependency")
    removed = int(mask_bad.sum())
    if removed:
        print(f"[{name}] uso de timestamp en etiqueta distinta a Timestamp-Dependency: -{removed:,}")
    return df[~mask_bad].reset_index(drop=True)


def run_validations(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ejecuta las validaciones sobre df. Comentar una línea desactiva esa validación."""
    df = check_starts_with_function(df, name)
    df = check_balanced_braces(df, name)
    df = check_txorigin_label(df, name)
    df = check_timestamp_label(df, name)
    return df


def main():
    df_sb = pd.read_csv(SMARTBUGS_CSV, encoding="utf-8")
    df_sl = pd.read_csv(SLITHER_CSV, encoding="utf-8")
    df_sol = pd.read_csv(SOLIDIFI_CSV, encoding="utf-8")
    print(f"SmartBugs: {len(df_sb):,} | Slither: {len(df_sl):,} | SolidiFI: {len(df_sol):,}")

    # Test = SolidiFI completo
    test_df = df_sol.reset_index(drop=True)

    # Dedup 1: eliminar de Slither lo que ya esta en SmartBugs
    sb_hashes = set(df_sb["function_code"].apply(code_hash))
    sl_hash = df_sl["function_code"].apply(code_hash)
    df_sl_dedup = df_sl[~sl_hash.isin(sb_hashes)]
    print(f"Dedup Slither vs SmartBugs: -{len(df_sl) - len(df_sl_dedup):,} funciones")

    # Dedup 2: eliminar de train lo que aparece en test (data leakage)
    train_df = pd.concat([df_sb, df_sl_dedup], ignore_index=True)
    test_hashes = set(test_df["function_code"].apply(code_hash))
    train_hash = train_df["function_code"].apply(code_hash)
    n_before = len(train_df)
    train_df = train_df[~train_hash.isin(test_hashes)].reset_index(drop=True)
    print(f"Dedup Test -> Train:        -{n_before - len(train_df):,} funciones")

    # Validaciones post-split (comentar una línea desactiva ese dataset)
    print("\nValidaciones:")
    train_df = run_validations(train_df, "TRAIN")
    test_df = run_validations(test_df, "TEST")

    train_df["id"] = range(len(train_df))
    test_df["id"] = range(len(test_df))
    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8")

    for name, df in (("TRAIN", train_df), ("TEST", test_df)):
        print(f"\n{name}: {len(df):,} funciones")
        print(df["vulnerability"].value_counts().to_string())
    print(f"\nGuardado: {TRAIN_CSV}\nGuardado: {TEST_CSV}")


if __name__ == "__main__":
    main()
