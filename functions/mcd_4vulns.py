import ast

import numpy as np
import pandas as pd

# ── Hiperparámetros ────────────────────────────────────────────────────────────
T = 0.8
MU_C = 0.1
EQUAL_RATE = 20.0

# ── Matriz de Capacidad de Detección (MCD) ────────────────────────────────────
MCD = {
    "honeybadger": {"reentrancy": 0, "time_manipulation": 0, "unchecked_low_calls": 0, "other": 67},
    "maian": {"reentrancy": 0, "time_manipulation": 0, "unchecked_low_calls": 0, "other": 0},
    "manticore": {"reentrancy": 25, "time_manipulation": 20, "unchecked_low_calls": 17, "other": 0},
    "mythril": {"reentrancy": 62, "time_manipulation": 0, "unchecked_low_calls": 42, "other": 0},
    "osiris": {"reentrancy": 62, "time_manipulation": 0, "unchecked_low_calls": 0, "other": 0},
    "oyente": {"reentrancy": 62, "time_manipulation": 0, "unchecked_low_calls": 0, "other": 0},
    "securify": {"reentrancy": 62, "time_manipulation": 0, "unchecked_low_calls": 25, "other": 0},
    "slither": {"reentrancy": 88, "time_manipulation": 40, "unchecked_low_calls": 33, "other": 100},
    "smartcheck": {
        "reentrancy": 62,
        "time_manipulation": 20,
        "unchecked_low_calls": 33,
        "other": 0,
    },
}

TOOLS = list(MCD.keys())
N_TOOLS = len(TOOLS)

VULN_MAP = {
    "Re-entrancy": "reentrancy",
    "Timestamp-Dependency": "time_manipulation",
    "Unhandled-Exception": "unchecked_low_calls",
    "tx.origin": "other",
}

VULN_CATS = list(VULN_MAP.keys())


# ── Helpers de Procesamiento ──────────────────────────────────────────────────
def _safe_get_tools_dict(tools_entry):
    """Convierte la columna tools a diccionario de forma segura."""
    if isinstance(tools_entry, dict):
        return tools_entry
    if isinstance(tools_entry, str):
        try:
            return ast.literal_eval(tools_entry)
        except Exception:
            return {}
    return {}


def _is_strictly_clean(tools_dict):
    """
    Verifica si ABSOLUTAMENTE NINGUNA herramienta detectó NADA
    (revisando el campo vulnerabilities de cada una).
    """
    if not tools_dict:
        return True

    for tool in tools_dict:
        # Si la herramienta tiene una lista de vulnerabilidades y no está vacía
        vulns = tools_dict[tool].get("vulnerabilities")
        if vulns and len(vulns) > 0:
            return False
    return True


# ── Funciones principales ──────────────────────────────────────────────────────
def compute_scores(details_str: str, tools_dict: dict, t: float = T) -> dict:
    scores = {cat: 0.0 for cat in VULN_CATS}

    # 1. Calcular scores para las vulnerabilidades detectadas
    if pd.notna(details_str) and str(details_str).strip() != "":
        detecciones = [d.strip().split(": ") for d in str(details_str).split(";")]
        for item in detecciones:
            if len(item) != 2:
                continue
            tool_name, vuln_name = item[0].lower(), item[1]

            if tool_name in MCD and vuln_name in VULN_MAP:
                mcd_key = VULN_MAP[vuln_name]
                rate = MCD[tool_name][mcd_key]
                scores[vuln_name] += np.exp(rate / (100.0 * t))

    # 2. Normalizar scores de vulnerabilidades
    for cat in VULN_CATS:
        scores[cat] = scores[cat] / N_TOOLS

    # 3. Lógica CLEAN ESTRICTA
    # Es clean (1.0) solo si pasó la verificación de _is_strictly_clean
    scores["clean"] = 1.0 if _is_strictly_clean(tools_dict) else 0.0

    return scores


def assign_labels(
    details_str: str, tools_dict: dict, mu_c: float = MU_C, t: float = T
) -> set | None:
    scores = compute_scores(details_str, tools_dict, t=t)

    # Aceptamos labels que superen el umbral
    accepted = {cat for cat, score in scores.items() if score >= mu_c}

    if not accepted:
        return None

    vuln_labels = accepted - {"clean"}
    # Prioridad: Si hay vulnerabilidades aceptadas por MCD, devolvemos esas.
    # Si no hay ninguna, pero 'clean' es 1.0, devolvemos {'clean'}.
    return vuln_labels if vuln_labels else ({"clean"} if scores["clean"] == 1.0 else None)


def apply_mcd_filter(
    df: pd.DataFrame,
    details_col: str = "detection_details",
    tools_col: str = "tools",
    t: float = T,
    mu_c: float = MU_C,
) -> pd.DataFrame:
    df = df.copy()

    # Procesar tools como diccionario una sola vez para ganar eficiencia
    df["tools_parsed"] = df[tools_col].apply(_safe_get_tools_dict)

    # Aplicar cálculos pasando ambos datos
    df["scores"] = df.apply(
        lambda row: compute_scores(row[details_col], row["tools_parsed"], t=t), axis=1
    )
    df["accepted_labels"] = df.apply(
        lambda row: assign_labels(row[details_col], row["tools_parsed"], mu_c=mu_c, t=t), axis=1
    )

    # Filtrado final
    n_before = len(df)
    df = df[df["accepted_labels"].notna()].reset_index(drop=True)
    n_discarded = n_before - len(df)

    print(f"Hiperparámetros usados: T={t}, μ_c={mu_c}")
    print(f"Contratos descartados (sin consenso suficiente): {n_discarded}")
    print(f"Contratos restantes: {len(df)}")

    return df.drop(columns=["tools_parsed"])  # Limpiamos la columna temporal
