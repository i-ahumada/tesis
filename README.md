# Aplicación de técnicas de Deep Learning para la detección automática de vulnerabilidades en Smart Contracts sobre tecnología blockchain Ethereum

## Set up

### 1. Entorno Python

```powershell
uv sync
```

### 2. Herramientas externas de preprocessing (slither + solc)

`slither-analyzer` y `solc-select` **no se instalan con `uv sync`** — se instalan globalmente con pip porque son herramientas CLI que deben estar en el PATH del sistema.

```powershell
pip install slither-analyzer==0.11.5 solc-select
```

Luego instalar todas las versiones de solc necesarias para el preprocessing:

```powershell
solc-select install 0.4.18 0.4.19 0.4.21 0.4.23 0.4.24 0.4.25 0.4.26 `
    0.5.0 0.5.16 0.5.17 `
    0.6.0 0.6.2 0.6.6 0.6.7 0.6.11 0.6.12 `
    0.7.0 0.7.4 0.7.5 0.7.6 `
    0.8.0 0.8.2 0.8.3 0.8.4 0.8.6 0.8.7 0.8.9 0.8.10 0.8.11 0.8.12 0.8.13 0.8.17
```

# 3. Uso de linter ruff
1. Ver qué hay que arreglar (solo reporta, no toca nada):
uv run ruff check . > errores_ruff.txt

2. Arreglar lo que puede arreglarse solo (imports desordenados, unused imports obvios, etc.):
uv run ruff check . --fix

3. Formatear el código (indentación, espacios, comillas, etc.):
uv run ruff format .

