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
    5. check_token_length: elimina snippets cuya tokenización con el
       tokenizador de CodeBERT (microsoft/codebert-base, sin tokens
       especiales) supera MAX_TOKENS (510 = 512 posiciones de CodeBERT
       menos los 2 tokens especiales CLS/SEP).

Balanceo de train (balance_dataset, aplicado solo a train — test es el
benchmark SolidiFI completo y no tiene columna `detector`):
- DROP_VULN_CAPS define, por categoría, un cap objetivo de filas tras el
  balanceo (no una cantidad fija a restar): cada categoría en el dict se
  recorta a ese cap. La cantidad a eliminar (drop_count = filas_actuales -
  cap) se reparte entre los detectores de esa categoría (columna `detector`,
  ver DETECTORS_BY_VULN en makeSlither.py/makeSmartbugs.py) de forma
  proporcional a la cantidad de apariciones de cada uno: un detector con más
  filas pierde una cuota mayor (reparto por método de mayor resto, para que
  la suma de cuotas sea exactamente drop_count). Si una categoría ya está en
  o por debajo de su cap no se elimina nada. Si un detector no tiene filas
  suficientes para cubrir su cuota, se eliminan las disponibles sin
  redistribuir el resto (el total dropeado puede quedar por debajo del
  objetivo).
- Categorías fuera de DROP_VULN_CAPS (ej. Unhandled-Exceptions, tx.origin) no
  se tocan: son las categorías minoritarias, recortarlas empeoraría el
  desbalance en vez de mejorarlo.

Ofuscación de train (obfuscate_dataset, paso final, solo train — test es el
benchmark SolidiFI y debe quedar intacto). No usa Slither: un snippet aislado
(solo la función, sin el contrato que la rodea) casi nunca compila por sí solo
—referencia variables de estado, eventos, modifiers o tipos declarados fuera
del snippet— y Slither/solc exigen compilar para construir su AST, así que
depender de Slither por snippet habría fallado en la mayoría de las filas
(confirmado empíricamente: un snippet con una sola referencia externa ya
rompe la compilación). En su lugar, se usan regex sobre el texto (sin
compilar), en este orden por fila:
    1. _obfuscate_strings: reemplaza el contenido de cada string literal por
       texto aleatorio de la misma longitud (conserva las comillas).
    2. _obfuscate_addresses: reemplaza literales hex de exactamente 40
       caracteres (0x + 40 hex, formato de address) por una dirección dummy
       aleatoria de 40 hex chars. No toca hex literals de otra longitud
       (bytes4/bytes32).
    3. _rename_identifiers: detecta declaraciones (nombre de la función o
       modifier, parámetros, variables locales, incluyendo named returns y
       declaraciones dentro de tuplas/for) vía el patrón sintáctico "dos
       identificadores consecutivos seguidos de = ; , ) o ]" — patrón que en
       Solidity solo ocurre en declaraciones, nunca en expresiones. No
       requiere resolver tipos: renombra params/locals/nombre de función de
       forma consistente dentro de la función (todas sus apariciones), sin
       tocar accesos a miembros (`.sender`, `.value`, etc., protegidos con un
       lookbehind de '.') ni las claves de opciones con nombre como
       {value: x} (protegidas por posición: precedidas de '{'/',' y seguidas
       de ':'). Los nombres nuevos se generan con un random.Random propio por
       fila (semilla RANDOM_SEED + índice de fila), por lo que el esquema de
       renombrado no se repite entre snippets.
   Al operar por regex (sin compilar), corre en segundos sobre todo el
   dataset. Es una aproximación best-effort: puede no detectar toda
   declaración (ej. `uint a, b;` sin repetir tipo) y ocasionalmente puede
   ofuscar de más una referencia a una variable de estado sin distinguirla de
   una local — ninguno de los dos casos rompe la sintaxis ni es un problema
   real para el objetivo (texto de entrenamiento, no código que deba volver a
   compilar).

Uso: python makeSplit.py  (requiere haber corrido antes los tres makeX.py)
"""

import hashlib
import random
import re
import string
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

_HERE = Path(__file__).parent
SMARTBUGS_CSV = _HERE.parent / "smartbugs" / "smartbugs_functions.csv"
SLITHER_CSV = _HERE.parent / "slither" / "slither_functions.csv"
SOLIDIFI_CSV = _HERE.parent / "solidifi" / "solidifi_functions.csv"
TRAIN_CSV = _HERE / "train_functions.csv"
TEST_CSV = _HERE / "test_functions.csv"

CODEBERT_MODEL = "microsoft/codebert-base"
MAX_TOKENS = 510  # 512 posiciones de CodeBERT - 2 tokens especiales (CLS/SEP)

RANDOM_SEED = 42
# Cap objetivo de filas por categoria tras el balanceo (ajustar segun la
# distribucion real). Solo incluye categorias mayoritarias: las minoritarias
# (Unhandled-Exceptions, tx.origin) no deben recortarse.
DROP_VULN_CAPS = {
    "Re-entrancy": 3500,
    "Timestamp-Dependency": 3500,
}


# ---------------------------------------------------------------------------
# Ofuscacion de train (ver docstring del modulo para el porque no usa Slither)
# ---------------------------------------------------------------------------

# Palabras reservadas / builtins de Solidity: nunca se renombran (ni como
# nombre nuevo ni como candidato detectado), para no corromper sintaxis fija
# (memory/storage/calldata, msg/block/tx, tipos elementales, etc.).
_SOL_RESERVED = {
    "pragma",
    "solidity",
    "import",
    "as",
    "from",
    "using",
    "is",
    "contract",
    "interface",
    "library",
    "abstract",
    "struct",
    "enum",
    "event",
    "error",
    "modifier",
    "function",
    "constructor",
    "fallback",
    "receive",
    "returns",
    "return",
    "if",
    "else",
    "for",
    "while",
    "do",
    "break",
    "continue",
    "throw",
    "emit",
    "revert",
    "require",
    "assert",
    "delete",
    "new",
    "try",
    "catch",
    "unchecked",
    "assembly",
    "public",
    "private",
    "internal",
    "external",
    "view",
    "pure",
    "payable",
    "virtual",
    "override",
    "constant",
    "immutable",
    "indexed",
    "anonymous",
    "memory",
    "storage",
    "calldata",
    "true",
    "false",
    "this",
    "super",
    "msg",
    "block",
    "tx",
    "now",
    "abi",
    "type",
    "gasleft",
    "selfdestruct",
    "suicide",
    "keccak256",
    "sha256",
    "ripemd160",
    "ecrecover",
    "addmod",
    "mulmod",
    "blockhash",
    "wei",
    "gwei",
    "ether",
    "seconds",
    "minutes",
    "hours",
    "days",
    "weeks",
    "years",
    "let",
    "_",
    "bool",
    "string",
    "bytes",
    "address",
    "int",
    "uint",
    "fixed",
    "ufixed",
    "var",
    "mapping",
}
_SOL_RESERVED |= {f"uint{n}" for n in range(8, 257, 8)}
_SOL_RESERVED |= {f"int{n}" for n in range(8, 257, 8)}
_SOL_RESERVED |= {f"bytes{n}" for n in range(1, 33)}

_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_ADDR_RE = re.compile(r"(?<![0-9a-fA-F])0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
# Declaracion = "tipo [ubicacion] nombre" seguido de = ; , ) o ] — patron que
# en Solidity solo aparece en declaraciones (parametros, locales, named
# returns, tuplas, for), nunca en una expresion. No requiere saber si el
# primer identificador es un tipo real: en el peor caso detecta de mas
# (inofensivo, ver docstring del modulo).
_DECL_CANDIDATE_RE = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)\b"
    r"(?:\s*\[[^\[\]]*\]\s*)*"
    r"\s+(?:(?:memory|storage|calldata)\s+)?"
    r"([A-Za-z_$][A-Za-z0-9_$]*)\b"
    r"\s*(?=[=;,)\]])"
)
_FUNC_NAME_RE = re.compile(r"\b(?:function|modifier)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")


def _random_token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}{rng.getrandbits(32):08x}"


def _obfuscate_strings(code: str, rng: random.Random) -> str:
    """Reemplaza el contenido de cada string literal por texto aleatorio de
    la misma longitud, conservando las comillas."""

    def _rand_content(m: re.Match) -> str:
        quote = m.group(0)[0]
        length = len(m.group(0)) - 2
        body = "".join(rng.choices(string.ascii_lowercase + string.digits, k=length))
        return f"{quote}{body}{quote}"

    return _STRING_RE.sub(_rand_content, code)


def _obfuscate_addresses(code: str, rng: random.Random) -> str:
    """Reemplaza literales hex de 40 caracteres (formato address) por una
    direccion dummy aleatoria de 40 hex chars."""
    return _ADDR_RE.sub(lambda m: "0x" + "".join(rng.choices("0123456789abcdef", k=40)), code)


def _rename_identifiers(code: str, rng: random.Random) -> str:
    """Renombra de forma consistente, dentro de la funcion, el nombre de la
    funcion/modifier y los parametros/variables locales detectados por
    _DECL_CANDIDATE_RE/_FUNC_NAME_RE. Protege accesos a miembros (`.sender`)
    y claves de opciones con nombre (`{value: x}`)."""
    mapping: dict = {}
    used = set()

    def new_name(prefix: str) -> str:
        while True:
            candidate = _random_token(rng, prefix)
            if candidate not in used:
                used.add(candidate)
                return candidate

    for m in _FUNC_NAME_RE.finditer(code):
        orig = m.group(1)
        if orig not in mapping and orig not in _SOL_RESERVED:
            mapping[orig] = new_name("f_")

    for m in _DECL_CANDIDATE_RE.finditer(code):
        orig = m.group(2)
        if orig not in mapping and orig not in _SOL_RESERVED:
            mapping[orig] = new_name("v_")

    if not mapping:
        return code

    pattern = re.compile(
        r"(?<!\.)\b(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")\b"
    )

    def repl(m: re.Match) -> str:
        name = m.group(1)
        before = code[: m.start()].rstrip()
        after = code[m.end() :].lstrip()
        if before.endswith(("{", ",")) and after.startswith(":"):
            return name  # clave de opcion con nombre (ej. {value: x}), no renombrar
        return mapping[name]

    return pattern.sub(repl, code)


def obfuscate_function_code(code: str, seed_key) -> str:
    """Aplica, en orden, ofuscacion de strings, direcciones e identificadores
    a un snippet. seed_key determina el RNG (reproducible por fila)."""
    rng = random.Random(seed_key)
    code = _obfuscate_strings(code, rng)
    code = _obfuscate_addresses(code, rng)
    code = _rename_identifiers(code, rng)
    return code


def obfuscate_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ofusca function_code de todo el DataFrame; una semilla distinta y
    reproducible por fila (RANDOM_SEED + posicion)."""
    df = df.copy()
    df["function_code"] = [
        obfuscate_function_code(code, f"{RANDOM_SEED}-{i}") for i, code in enumerate(df["function_code"])
    ]
    print(f"[{name}] ofuscadas {len(df):,} funciones (identificadores, strings, direcciones)")
    return df


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


def check_token_length(df: pd.DataFrame, name: str, tokenizer) -> pd.DataFrame:
    """Elimina snippets cuya tokenización con el tokenizador de CodeBERT (sin
    tokens especiales) supera MAX_TOKENS."""
    counts = df["function_code"].apply(lambda c: len(tokenizer.encode(str(c), add_special_tokens=False)))
    mask_ok = counts <= MAX_TOKENS
    removed = int((~mask_ok).sum())
    if removed:
        print(f"[{name}] más de {MAX_TOKENS} tokens: -{removed:,}")
    return df[mask_ok].reset_index(drop=True)


def run_validations(df: pd.DataFrame, name: str, tokenizer) -> pd.DataFrame:
    """Ejecuta las validaciones sobre df. Comentar una línea desactiva esa validación."""
    df = check_starts_with_function(df, name)
    df = check_balanced_braces(df, name)
    df = check_txorigin_label(df, name)
    df = check_timestamp_label(df, name)
    df = check_token_length(df, name, tokenizer)
    return df


# ---------------------------------------------------------------------------
# Balanceo de train por categoria/detector
# ---------------------------------------------------------------------------


def _proportional_quotas(counts: "pd.Series[int]", drop_count: int) -> dict:
    """Reparte drop_count entre los indices de `counts` en proporcion a su valor
    (mas apariciones -> cuota mayor). Metodo de mayor resto: trunca cada cuota
    y reparte las unidades sobrantes a quienes tenian el resto mas grande, para
    que la suma de cuotas de exactamente drop_count (salvo limite de filas
    disponibles, que se aplica despues)."""
    total = int(counts.sum())
    raw = {k: drop_count * v / total for k, v in counts.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    remainder = drop_count - sum(quotas.values())
    for k in sorted(raw, key=lambda k: raw[k] - quotas[k], reverse=True)[:remainder]:
        quotas[k] += 1
    return quotas


def balance_dataset(df: pd.DataFrame, caps: dict, name: str) -> pd.DataFrame:
    """Recorta cada categoria en `caps` a su cap objetivo de filas. Lo que
    sobra (filas_actuales - cap) se reparte entre los detectores de esa
    categoria (columna `detector`) en proporcion a su cantidad de apariciones:
    un detector con mas filas pierde una cuota mayor. Categorias ya por
    debajo de su cap no se tocan. Si un detector no tiene filas suficientes
    para cubrir su cuota, se eliminan las disponibles sin redistribuir el
    resto."""
    drop_idx = []
    for vuln, cap in caps.items():
        subset = df[df["vulnerability"] == vuln]
        counts = subset["detector"].value_counts()
        if counts.empty:
            continue
        total = int(counts.sum())
        drop_count = max(0, total - cap)
        if drop_count == 0:
            print(f"[{name}] {vuln}: {total:,} <= cap {cap:,}, no se elimina nada")
            continue
        quotas = _proportional_quotas(counts, drop_count)
        removed_total = 0
        for det in sorted(quotas):
            det_rows = subset[subset["detector"] == det]
            n = min(quotas[det], len(det_rows))
            if n:
                drop_idx.extend(det_rows.sample(n=n, random_state=RANDOM_SEED).index.tolist())
            removed_total += n
            print(f"[{name}] {vuln} / {det}: -{n:,} (de {len(det_rows):,})")
        print(f"[{name}] {vuln} total: -{removed_total:,} (cap {cap:,})")
    return df.drop(index=drop_idx).reset_index(drop=True)


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
    print("\nCargando tokenizador de CodeBERT...")
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)

    print("\nValidaciones:")
    train_df = run_validations(train_df, "TRAIN", tokenizer)
    test_df = run_validations(test_df, "TEST", tokenizer)

    # Balanceo de train (test = SolidiFI completo, no se toca)
    print("\nBalanceo:")
    train_df = balance_dataset(train_df, DROP_VULN_CAPS, "TRAIN")

    # Ofuscacion de train, paso final antes de guardar (test no se toca)
    print("\nOfuscación:")
    train_df = obfuscate_dataset(train_df, "TRAIN")

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
