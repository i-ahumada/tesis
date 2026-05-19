import pandas as pd
from transformers import PreTrainedTokenizerBase


def count_tokens(code: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Retorna la cantidad de tokens que produce el tokenizador para un código."""
    return len(tokenizer.encode(code, add_special_tokens=True, truncation=False))


def filter_by_token_length(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 512,
    code_col: str = "source_code",
) -> pd.DataFrame:
    """
    Descarta contratos cuyo source_code supere max_tokens tokens.
    Agrega columna 'token_count' al DataFrame.

    Returns:
        DataFrame filtrado (contratos con token_count <= max_tokens).
    """
    df = df.copy()
    print("Calculando longitudes de tokenización...")
    df["token_count"] = df[code_col].apply(lambda c: count_tokens(c, tokenizer))

    n_before = len(df)
    df = df[df["token_count"] <= max_tokens].reset_index(drop=True)
    n_removed = n_before - len(df)

    print(
        f"Contratos descartados (> {max_tokens} tokens): {n_removed}  ({100 * n_removed / n_before:.1f}%)"
    )
    print(f"Contratos restantes                          : {len(df)}")
    print()
    print(df["token_count"].describe().to_string())

    return df


def truncate_by_token_length(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 512,
    code_col: str = "source_code",
) -> pd.DataFrame:
    """
    Trunca el source_code de cada contrato a max_tokens tokens.
    Agrega columna 'token_count' con el conteo original (pre-truncado).

    Returns:
        DataFrame con source_code truncado.
    """
    df = df.copy()
    print("Calculando longitudes de tokenización...")
    df["token_count"] = df[code_col].apply(lambda c: count_tokens(c, tokenizer))

    n_truncated = (df["token_count"] > max_tokens).sum()

    def _truncate(code: str) -> str:
        ids = tokenizer.encode(
            code, add_special_tokens=True, truncation=True, max_length=max_tokens
        )
        return tokenizer.decode(ids, skip_special_tokens=True)

    df[code_col] = df[code_col].apply(_truncate)

    print(
        f"Contratos truncados (> {max_tokens} tokens): {n_truncated}  ({100 * n_truncated / len(df):.1f}%)"
    )
    print(f"Total contratos                            : {len(df)}")
    print()
    print(df["token_count"].describe().to_string())

    return df
