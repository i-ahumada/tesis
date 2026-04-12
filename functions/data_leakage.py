import pandas as pd
from functions.deduplication import compute_md5
from functions.preprocessing import preprocess_source


def check_curated_leakage(
    df: pd.DataFrame,
    curated_path: str,
    code_col: str = 'source_code',
    hash_col: str = 'md5_hash',
) -> pd.DataFrame:
    """
    Remueve contratos de df que aparezcan en Smartbugs Curated para evitar
    data leakage. Solo compara source_code (ya normalizado) via MD5.

    Args:
        df:           DataFrame principal (Smartbugs Wild).
        curated_path: Path al CSV de Smartbugs Curated
                      (debe tener columna source_code).
        code_col:     Nombre de la columna de código en df.
        hash_col:     Nombre de la columna MD5 en df (debe existir).

    Returns:
        DataFrame filtrado sin contratos presentes en Curated.
    """
    # TODO: Descomentar y llamar esta función cuando el dataset esté disponible.
    # df_curated = pd.read_csv(curated_path)
    # curated_hashes = set(
    #     df_curated[code_col].apply(preprocess_source).apply(compute_md5)
    # )
    # n_before = len(df)
    # df = df[~df[hash_col].isin(curated_hashes)].reset_index(drop=True)
    # print(f'Contratos de Curated removidos (data leakage): {n_before - len(df)}')
    # print(f'Contratos restantes                          : {len(df)}')
    # return df

    print('Data leakage check: PENDIENTE — Smartbugs Curated no disponible.')
    return df
