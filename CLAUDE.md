# Reglas del proyecto (tesis)

Proyecto de tesis: detección de vulnerabilidades en smart contracts (Solidity)
con CodeBERT. El pipeline de preprocesamiento vive en `preprocessing/`.

## Formato y linting

- El linter/formatter es **ruff** (configurado en `pyproject.toml`, line-length 120).
- Un hook de `PostToolUse` corre `ruff format` + `ruff check --fix` automáticamente
  cada vez que se escribe o edita un archivo. No hace falta correrlo a mano.

## Comentario de decisiones de diseño en archivos `.py`

Cada archivo `.py` **debe** tener, arriba de todo (antes de los imports), un
comentario multilínea (docstring de módulo con `"""..."""`) que explique las
**decisiones de diseño** tomadas en ese archivo: qué hace y en qué orden.

Reglas para mantener ese comentario:

- **Cada vez que se modifica un archivo `.py`, releer el código completo del
  archivo** y usar su contenido como contexto para (re)escribir el docstring de
  módulo, de modo que siga reflejando fielmente qué hace el archivo y en qué
  orden ocurren los pasos principales.
- El docstring debe describir el flujo (los pasos y su orden), no repetir el
  código línea por línea.
- Si el archivo aún no tiene este docstring, agregarlo al crearlo o al primera
  modificación.
- Debe ir siempre como primera sentencia del módulo (encima de los imports),
  para que ruff (regla E402) y la convención de docstrings de módulo lo respeten.

Ejemplo de forma esperada:

```python
"""
Genera el dataset de Slither.

Decisiones de diseño y orden de ejecución:
1. Normaliza el código fuente de cada contrato (...).
2. Corre Slither por contrato detectando la versión de solc (...).
3. Extrae, por cada detección, el rango de líneas de la función afectada (...).
4. Escribe las filas resultantes a `slither_functions.csv`.
"""

import ...
```
