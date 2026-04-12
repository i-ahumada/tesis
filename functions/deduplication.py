import hashlib
import pandas as pd


def compute_md5(code: str) -> str:
    """Calcula el hash MD5 de un string de código fuente."""
    return hashlib.md5(code.encode('utf-8')).hexdigest()


def deduplicate_df(df: pd.DataFrame, code_col: str = 'source_code') -> pd.DataFrame:
    """
    Agrega columna md5_hash y elimina filas con código duplicado.
    Retorna el DataFrame deduplicado y reseteado.
    """
    df = df.copy()
    df['md5_hash'] = df[code_col].apply(compute_md5)

    n_before = len(df)
    df = df.drop_duplicates(subset='md5_hash').reset_index(drop=True)
    n_removed = n_before - len(df)

    print(f'Contratos antes de deduplicación : {n_before}')
    print(f'Duplicados eliminados            : {n_removed}')
    print(f'Contratos restantes              : {len(df)}')

    return df
