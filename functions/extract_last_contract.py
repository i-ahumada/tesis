#########################################################
### FUNCIÓN: MINIFICACIÓN DE CÓDIGO SOLIDITY ###
#########################################################

# EJEMPLO DE USO: df['source_code'] = df['source_code'].apply(clean_extract_and_minify)

import re

"""
TODO
Elimina comentarios para identificar correctamente el último contrato.
Armar otra opción donde no sea necesario eliminar comentarios ya que pueden poseer información valiosa.
"""


def clean_extract_and_minify(code: str) -> str:
    """
    Realiza una limpieza del código fuente para eliminar código de librerias.

    Objetivos principales:
    1. Enfoque de Lógica: Extrae solo el último contrato del archivo (generalmente
       el contrato principal).
    2. Optimización de Tokens: La minificación agresiva permite que una mayor
       cantidad de lógica de control quepa dentro del límite de 512 tokens de CodeBERT.

    Pasos:
    - Regex 1 & 2: Remoción de comentarios multilínea (/* */) y unilínea (//).
    - Regex 3: Identifica todos los 'contract Name' y recorta el string desde el inicio del último.

    Args:
        code (str): Código fuente original de Solidity.

    Returns:
        str: Código minificado.
    """
    if not isinstance(code, str):
        return ""

    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)

    matches = list(re.finditer(r"\bcontract\s+(\w+)", code))
    if matches:
        last_contract_start = matches[-1].start()
        code = code[last_contract_start:]

    return code.strip()
