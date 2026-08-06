# Aplicación de IA para la detección de vulnerabilidades en smart contracts sobre tecnología blockchain Ethereum

### Application of AI for the detection of vulnerabilities in smart contracts on Ethereum blockchain technology

**Autores:** Iván A. Ahumada\*, Franco E. Morales

Departamento de Ingeniería en Informática, Facultad de Ingeniería, Universidad Nacional de Mar del Plata, Mar del Plata, Argentina.

\*Correo electrónico de contacto: ahumadaivan@alumnos.fi.mdp.edu.ar

---

## Resumen

Este trabajo aborda la detección automática de vulnerabilidades en *smart contracts* de Solidity mediante la aplicación de inteligencia artificial para superar las limitaciones de otras técnicas. El ecosistema involucra activos valuados en millones de dólares en tecnología *blockchain* como Ethereum.

Se usó como base un modelo de *deep learning* basado en CodeBERT; además de agregarle una capa de clasificación, se le realizó *fine-tuning* y optimización de hiperparámetros. Para el desarrollo se crearon conjuntos de datos etiquetados de funciones vulnerables para entrenamiento partiendo de los bancos de datos públicos *Slither Audited Smart Contract Dataset* y *Smartbugs Wild*. Mientras que para la evaluación del modelo se utilizó el *dataset* SolidiFi Benchmark, el cual es representativo como *benchmark* de la industria para verificar herramientas de análisis de vulnerabilidades. Se tomó un subconjunto de cuatro fallas de seguridad representativas de la problemática con el fin de obtener resultados comparables con la literatura. Estas son: *Reentrancy*, *Timestamp Dependency*, *Unhandled Exception* y *tx.origin*.

La evaluación sobre el *benchmark* dio como resultado métricas competitivas; un F1-macro de 84,11 % y una precisión del 92,33 %, resultados competitivos respecto de otras soluciones actuales. El modelo evidencia ser un buen complemento para herramientas de análisis estático como Slither en la detección de fallas. Además, los *datasets* generados y el modelo son de código abierto, alineándose con la filosofía de transparencia y la naturaleza abierta del ecosistema *blockchain*.

Los resultados obtenidos demuestran el lugar que tienen las técnicas de *deep learning* para la detección de ciberamenazas propias de programas en entornos descentralizados. Pone en evidencia la capacidad que tienen los modelos de inteligencia artificial de mejorar las reglas rígidas de los métodos clásicos y contribuye al conjunto de herramientas de ciberdefensa que participan del continuo esfuerzo de formación de una infraestructura financiera digital segura.

***Palabras clave:*** Blockchain, IA, Deep Learning, Detección de Vulnerabilidades, Detección de Ciberamenazas

***Simposios:*** Capacidades Tecnológicas.

---

## Abstract

This work addresses the automatic detection of vulnerabilities in Solidity smart contracts through the application of artificial intelligence to overcome the limitations of other techniques. The ecosystem involves assets valued at millions of dollars in blockchain technology such as Ethereum.

A deep learning model based on CodeBERT was used as a foundation; in addition to adding a classification layer, fine-tuning and hyperparameter optimization were performed. For development, labeled datasets of vulnerable functions were created for training, based on the public databases *Slither Audited Smart Contract Dataset* and *Smartbugs Wild*. For model evaluation, the *SolidiFi Benchmark* dataset was used, which is representative as an industry benchmark for verifying vulnerability analysis tools. A subset of four security flaws representative of the problem was selected in order to obtain results comparable with the literature. These are: *Reentrancy*, *Timestamp Dependency*, *Unhandled Exception*, and *tx.origin*.

Evaluation on the benchmark yielded competitive metrics: a macro F1-score of 84.11 % and an accuracy of 92.33 %, results that are competitive with respect to other current solutions. The model proves to be a good complement to static analysis tools such as Slither in detecting flaws. Furthermore, the generated datasets and the model itself are open source, aligning with the philosophy of transparency and the open nature of the blockchain ecosystem.

The results obtained demonstrate the role that deep learning techniques play in detecting cyber threats specific to programs running in decentralized environments. They highlight the capacity of artificial intelligence models to improve upon the rigid rules of classical methods, contributing to the toolkit of cyberdefense techniques involved in the ongoing effort to build a secure digital financial infrastructure.

***Keywords:*** Blockchain, AI, Deep Learning, Vulnerability Detection, Cyber Threat Detection

***Symposia:*** Technological Capabilities.

---

## Estructura del repositorio

```
.
├── preprocessing/        # Scripts para generar los datasets etiquetados
│   ├── slither/          #   makeSlither.py  → Slither Audited SC Dataset
│   ├── smartbugs/        #   makeSmartbugs.py → Smartbugs Wild
│   ├── solidifi/         #   makeSolidifi.py  → SolidiFi Benchmark (test)
│   └── split/            #   makeSplit.py     → train/test + deduplicación
├── functions/            # Utilidades compartidas (normalización, tokenización,
│                         #   deduplicación, detección de data leakage, stats, MCD)
├── datasets/             # Datasets generados (ver datasets/v1.0/README.md)
│   └── v1.0/             #   *_functions.csv + train/test + notas de generación
├── notebooks/            # Notebooks de entrenamiento, fine-tuning y evaluación
├── context/              # Material de referencia (paper.pdf, etc.)
├── pyproject.toml        # Configuración del proyecto y de ruff (uv)
├── uv.lock               # Lockfile de dependencias
├── CLAUDE.md             # Reglas del repo para Claude Code
└── LICENSE               # Licencia MIT
```

---

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

### 3. Linter / formatter (ruff)

El proyecto usa **ruff** (configurado en `pyproject.toml`). Comandos manuales:

```powershell
# 1. Ver qué hay que arreglar (solo reporta, no toca nada):
uv run ruff check . > errores_ruff.txt

# 2. Arreglar lo auto-corregible (imports desordenados, unused imports, etc.):
uv run ruff check . --fix

# 3. Formatear el código (indentación, espacios, comillas, etc.):
uv run ruff format .
```

---

## Reproducir el pipeline

El pipeline genera los datasets etiquetados a partir de las fuentes públicas y
produce los conjuntos de entrenamiento y test. Ejecutar en este orden (requiere
haber completado el *Set up*, incluidos `slither-analyzer` y `solc-select`):

```powershell
# 1. Datasets de entrenamiento (etiquetado con Slither)
python preprocessing/slither/makeSlither.py       # Slither Audited SC Dataset
python preprocessing/smartbugs/makeSmartbugs.py   # Smartbugs Wild

# 2. Dataset de test (etiquetado por nombre de función inyectada)
python preprocessing/solidifi/makeSolidifi.py     # SolidiFi Benchmark

# 3. Split final: unión, deduplicación y prevención de data leakage
python preprocessing/split/makeSplit.py           # → train/test en datasets/
```

<!-- TODO: agregar el/los notebook(s) de entrenamiento/fine-tuning y evaluación,
     y el comando o los pasos para reproducir el entrenamiento del modelo. -->

---

## Datasets

Los datasets etiquetados de funciones (una fila por función, con su
vulnerabilidad) se encuentran en [`datasets/*/`](datasets/v1.0/). Ver
[`datasets/*/README.md`](datasets/v1.0/README.md) para el detalle de
generación, detectores usados y deduplicación.

**Fuentes públicas:**

- **Slither Audited Smart Contract Dataset** — entrenamiento.
- **Smartbugs Wild** — entrenamiento.
- **SolidiFi Benchmark** — evaluación (test).

**Vulnerabilidades consideradas:** *Reentrancy*, *Timestamp Dependency*,
*Unhandled Exception* y *tx.origin*.

---

## Resultados

Evaluación sobre el *SolidiFi Benchmark*:

| Métrica           | Valor    |
|-------------------|---------:|
| F1-macro          | 84,11 %  |
| Precisión (accuracy) | 92,33 % |

<!-- TODO: agregar la tabla de métricas por clase (precision/recall/F1 por
     vulnerabilidad) y, si corresponde, la comparación con Slither u otras
     soluciones de la literatura. -->

---

## Modelo

El modelo se basa en **CodeBERT** con una capa de clasificación añadida,
*fine-tuning* y optimización de hiperparámetros.

<!-- TODO: agregar el enlace al modelo entrenado (por ejemplo, Hugging Face Hub
     o un release de GitHub) y un ejemplo mínimo de inferencia:

```python
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("<usuario/modelo>")
# model = AutoModelForSequenceClassification.from_pretrained("<usuario/modelo>")
# ...
```
-->

---

## Cita

Si utilizás este trabajo, los datasets o el modelo, por favor citá:

<!-- TODO: completar los datos de la publicación (evento/revista, año, páginas, DOI). -->

```bibtex
@inproceedings{ahumada_ai_smartcontracts,
  title     = {Aplicación de IA para la detección de vulnerabilidades en smart
               contracts sobre tecnología blockchain Ethereum},
  author    = {Ahumada, Iván A. and Morales, Franco E. and Hinojal, Hernán and
               Seijas, Leticia M.},
  year      = {2026},
  note      = {Universidad Nacional de Mar del Plata}
}
```

---

## Licencia

Este proyecto se distribuye bajo la licencia **MIT** — ver el archivo
[`LICENSE`](LICENSE) para el texto completo.
