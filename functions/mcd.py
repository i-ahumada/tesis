import numpy as np
import pandas as pd

# CUIDADO: Modifiqué como se determinan los contratos "clean" si no mal recuerdo.
# TODO:  Revisar después para dejarlo como estaba originalmente.

# ── Hiperparámetros ────────────────────────────────────────────────────────────
T = 0.8  # Temperatura: a mayor T, menor diferencia entre herramientas buenas y malas
MU_C = 0.1  # Umbral de aceptación de un label (permisivo)

# Tasa uniforme para categorías donde ninguna herramienta tiene accuracy medida.
# Permite que los tools que detectan esa categoría contribuyan con igual peso.
EQUAL_RATE = 20.0

# ── Matriz de Capacidad de Detección (MCD) ────────────────────────────────────
# Fuente: SmartBugs paper — accuracy de cada herramienta por categoría (%)
# Filas: herramientas | Columnas: categorías de vulnerabilidad
MCD = {
    #                        ac    arith  ds    fr    re    tm    ucl   other
    "honeybadger": {
        "access_control": 0,
        "arithmetic": 0,
        "denial_service": 0,
        "front_running": 0,
        "reentrancy": 0,
        "time_manipulation": 0,
        "unchecked_low_calls": 0,
        "other": 67,
    },
    "maian": {
        "access_control": 0,
        "arithmetic": 0,
        "denial_service": 0,
        "front_running": 0,
        "reentrancy": 0,
        "time_manipulation": 0,
        "unchecked_low_calls": 0,
        "other": 0,
    },
    "manticore": {
        "access_control": 21,
        "arithmetic": 18,
        "denial_service": 0,
        "front_running": 0,
        "reentrancy": 25,
        "time_manipulation": 20,
        "unchecked_low_calls": 17,
        "other": 0,
    },
    "mythril": {
        "access_control": 21,
        "arithmetic": 68,
        "denial_service": 0,
        "front_running": 29,
        "reentrancy": 62,
        "time_manipulation": 0,
        "unchecked_low_calls": 42,
        "other": 0,
    },
    "osiris": {
        "access_control": 0,
        "arithmetic": 50,
        "denial_service": 0,
        "front_running": 0,
        "reentrancy": 62,
        "time_manipulation": 0,
        "unchecked_low_calls": 0,
        "other": 0,
    },
    "oyente": {
        "access_control": 0,
        "arithmetic": 55,
        "denial_service": 0,
        "front_running": 0,
        "reentrancy": 62,
        "time_manipulation": 0,
        "unchecked_low_calls": 0,
        "other": 0,
    },
    "securify": {
        "access_control": 0,
        "arithmetic": 0,
        "denial_service": 0,
        "front_running": 29,
        "reentrancy": 62,
        "time_manipulation": 0,
        "unchecked_low_calls": 25,
        "other": 0,
    },
    "slither": {
        "access_control": 21,
        "arithmetic": 0,
        "denial_service": 0,
        "front_running": 0,
        "reentrancy": 88,
        "time_manipulation": 40,
        "unchecked_low_calls": 33,
        "other": 100,
    },
    "smartcheck": {
        "access_control": 11,
        "arithmetic": 5,
        "denial_service": 0,
        "front_running": 0,
        "reentrancy": 62,
        "time_manipulation": 20,
        "unchecked_low_calls": 33,
        "other": 0,
    },
}

TOOLS = list(MCD.keys())
VULN_CATS = [
    "access_control",
    "arithmetic",
    "denial_service",
    "front_running",
    "reentrancy",
    "time_manipulation",
    "unchecked_low_calls",
    "other",
]
N_TOOLS = len(TOOLS)

# Categorías donde TODOS los tools tienen accuracy 0 → se aplica EQUAL_RATE
EQUAL_WEIGHT_CATS = {cat for cat in VULN_CATS if all(MCD[tool][cat] == 0 for tool in TOOLS)}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _normalize_cat(name: str) -> str:
    """Normaliza nombres de categoría al formato del MCD (lower + underscores)."""
    return name.lower().replace(" ", "_").replace("-", "_")


# ── Funciones principales ──────────────────────────────────────────────────────
def compute_scores(tools_dict: dict, t: float = T) -> dict:
    """
    Calcula score_c para cada categoría de vulnerabilidad y para 'clean'.

    Vulnerabilidades:
        score_c = Σ_{i ∈ detectaron c} exp(rate_{i,c} / (100·t)) / n_tools
        Para categorías en EQUAL_WEIGHT_CATS (rate 0% en todas las herramientas)
        se usa EQUAL_RATE como tasa fija, dando peso uniforme a cada tool.

    Clean:
        Un tool "detecta clean" si no reportó ninguna categoría.
        score_clean = (tools que no detectaron nada) / n_tools
        Cada tool contribuye con 1/n_tools — peso igual para todos.
        El label igual debe superar MU_C para ser aceptado.

    Returns:
        dict {categoria: score}  — incluye 'clean'
    """
    scores = {}

    for cat in VULN_CATS:
        acc = 0.0
        for tool in TOOLS:
            detected = {
                _normalize_cat(k) for k in tools_dict.get(tool, {}).get("categories", {}).keys()
            }
            if cat in detected:
                rate = EQUAL_RATE if cat in EQUAL_WEIGHT_CATS else MCD[tool][cat]
                acc += np.exp(rate / (100.0 * t))
        scores[cat] = acc / N_TOOLS

    n_clean_votes = sum(1 for tool in TOOLS if not tools_dict.get(tool, {}).get("categories", {}))
    scores["clean"] = 1.0 if n_clean_votes == N_TOOLS else 0.0

    return scores


def assign_labels(tools_dict: dict, mu_c: float = MU_C, t: float = T) -> set | None:
    """
    Retorna el conjunto de labels aceptados para un contrato, o None si se
    descarta (ningún label, incluido 'clean', supera mu_c).

    Si hay categorías de vulnerabilidad aceptadas, devuelve solo esas.
    Si solo 'clean' supera el umbral, devuelve {'clean'}.
    """
    scores = compute_scores(tools_dict, t=t)
    accepted = {cat for cat, score in scores.items() if score >= mu_c}

    if not accepted:
        return None

    vuln_labels = accepted - {"clean"}
    return vuln_labels if vuln_labels else {"clean"}


def apply_mcd_filter(
    df: pd.DataFrame,
    tools_col: str = "tools",
    t: float = T,
    mu_c: float = MU_C,
) -> pd.DataFrame:
    """
    Aplica compute_scores y assign_labels a todo el DataFrame.
    Descarta contratos donde assign_labels retorna None.

    Agrega columnas 'scores' y 'accepted_labels'.
    Retorna el DataFrame filtrado y reseteado.

    Parámetros sobreescribibles desde el notebook:
        t    — Temperatura (default: módulo T)
        mu_c — Umbral de aceptación (default: módulo MU_C)
    """
    df = df.copy()
    df["scores"] = df[tools_col].apply(lambda d: compute_scores(d, t=t))
    df["accepted_labels"] = df[tools_col].apply(lambda d: assign_labels(d, mu_c=mu_c, t=t))

    n_before = len(df)
    df = df[df["accepted_labels"].notna()].reset_index(drop=True)
    n_discarded = n_before - len(df)

    print(f"Hiperparámetros usados: T={t}, μ_c={mu_c}")
    print(f"Contratos descartados (ningún label ≥ μ_c={mu_c}): {n_discarded}")
    print(f"Contratos restantes                               : {len(df)}")

    return df
