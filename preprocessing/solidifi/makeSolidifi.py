"""
makeSolidifi.py — Extrae las funciones vulnerables del SolidiFI-benchmark.

Decisiones de diseño:
- Las funciones vulnerables se identifican por NOMBRE: SolidiFI inyecta funciones
  con sufijos característicos por categoría (_re_entN, _tmstmpN, _txoriginN,
  _unchkN/_uncheckN). El BugLog trae números de línea corridos, por eso no se usa.
- Solo se extraen funciones: de las 5.434 entradas del BugLog, 5.179 son funciones
  inyectadas; las 255 restantes son declaraciones de variables sueltas y se omiten.
- Normalización: se eliminan comentarios y se re-indenta con jsbeautifier.
- Sin dedup por hash: el mismo template se inyecta en muchos contratos y cada
  instancia es un caso de test válido.

Uso: python makeSolidifi.py
"""

import re
from pathlib import Path

import jsbeautifier
import pandas as pd

VULN_NAME_PATTERNS = {
    "Re-entrancy": re.compile(r"_re_ent\d+$"),
    "Timestamp-Dependency": re.compile(r"_tmstmp\d+$"),
    "Unhandled-Exceptions": re.compile(r"_(?:unchk|uncheck)\d+$"),
    "tx.origin": re.compile(r"_txorigin\d+$"),
}

_HERE = Path(__file__).parent
DATA_DIR = _HERE / "data" / "buggy_contracts"
OUTPUT_CSV = _HERE / "solidifi_functions.csv"

_FUNC_RE = re.compile(r"^\s*function\s+(\w+)")

_BEAUTIFY_OPTS = jsbeautifier.default_options()
_BEAUTIFY_OPTS.indent_size = 4


def strip_comments(source: str) -> str:
    """Elimina // y /* */ preservando los saltos de linea."""
    source = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def normalize_source(source: str) -> str:
    return strip_comments(jsbeautifier.beautify(source, _BEAUTIFY_OPTS))


def find_block_end(lines: list, start_idx: int):
    """Indice (0-based) de la linea que cierra el bloque {} iniciado en start_idx."""
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


def extract_injected_functions(source: str, name_pattern: re.Pattern) -> list:
    """Retorna [(nombre, codigo)] de las funciones cuyo nombre matchea el patron."""
    lines = source.splitlines()
    found = []
    for i, line in enumerate(lines):
        m = _FUNC_RE.match(line)
        if not m or not name_pattern.search(m.group(1)):
            continue
        end = find_block_end(lines, i)
        if end is None:
            continue
        found.append((m.group(1), "\n".join(ln for ln in lines[i : end + 1] if ln.strip())))
    return found


def main():
    rows = []
    for vuln, pattern in VULN_NAME_PATTERNS.items():
        vuln_dir = DATA_DIR / vuln
        n_before = len(rows)
        for sol_path in sorted(vuln_dir.glob("*.sol")):
            source = normalize_source(sol_path.read_text(encoding="utf-8", errors="ignore"))
            seen = set()
            for name, code in extract_injected_functions(source, pattern):
                if name in seen:
                    continue
                seen.add(name)
                rows.append(
                    {
                        "id": len(rows),
                        "contract_file": sol_path.name,
                        "vulnerability": vuln,
                        "function_name": name,
                        "function_code": code,
                        "dataset": "solidifi",
                    }
                )
        print(f"{vuln:<25} {len(rows) - n_before:>6,} funciones")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"{'TOTAL':<25} {len(df):>6,} funciones")
    print(f"Guardado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
