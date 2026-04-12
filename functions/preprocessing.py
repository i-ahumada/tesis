import re


def remove_comments(code: str) -> str:
    """Elimina comentarios de Solidity: bloques /* ... */ y líneas // ..."""
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//[^\n]*', '', code)
    return code


def normalize_whitespace(code: str) -> str:
    """Reemplaza \\t, \\n, \\r por espacio y colapsa espacios múltiples."""
    code = code.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    code = re.sub(r' {2,}', ' ', code)
    return code.strip()


def preprocess_source(code: str) -> str:
    """Pipeline completo: elimina comentarios y normaliza whitespace."""
    code = remove_comments(code)
    code = normalize_whitespace(code)
    return code
