# Decisiones de Preprocesamiento de Datasets

Referencia del paper base: Tang et al., *Scientific Reports* 2023, doi:10.1038/s41598-023-47219-0  
Vulnerabilidades objetivo: **Re-entrancy**, **Timestamp-Dependency**, **Unhandled-Exceptions**, **tx.origin**

---

## 1. Dataset SolidiFI (conjunto de TEST)

**Fuente:** `SolidiFI-benchmark/buggy_contracts/` — 50 contratos por tipo de vulnerabilidad con CSVs de BugLog  
**Script:** `func_dataset/makeSolidifi.py`  
**Output:** `func_dataset/solidifi/solidifi_functions.csv` — 5,117 funciones

### 1.1 Fuente primaria: BugLog (no Slither)

**Decisión:** Las funciones se extraen a partir del BugLog (CSV de ground truth del benchmark), no del output de Slither.  
**Justificación:** Slither no detecta el ~18% de los bugs inyectados (especialmente Timestamp-Dependency y Unhandled-Exceptions). BugLog es la fuente de verdad del benchmark — cada línea indica exactamente qué función fue modificada para inyectar la vulnerabilidad. Usar Slither como fuente primaria introduciría falsos negativos en el test set.  
**Rol de Slither en SolidiFI:** solo para estadísticas de validación (cobertura del detector).

### 1.2 Deduplicación: intra-contrato por tripleta

**Decisión:** Dedup por `(vuln, sol_file, func_name)`.  
**Justificación:** SolidiFI inyecta funciones con nombres únicos por tipo de vulnerabilidad (e.g., `callme_re_ent7`) en cada uno de los 50 contratos. La misma función aparece en múltiples contratos con cuerpos idénticos o muy similares pero representa instancias de test válidas e independientes. Aplicar hash dedup eliminaría instancias legítimas y reduciría el test set de 5,117 a ~652 — una pérdida del 87% de las muestras. El paper reporta 5,434 snippets de test; nuestra cifra (5,117) es consistente con eso.

### 1.3 NO se aplica hash dedup cross-contrato

**Decisión:** No se aplica SHA-256 dedup entre contratos de SolidiFI.  
**Justificación:** Las funciones inyectadas son plantillas (mismo código, misma lógica), pero representan instancias de vulnerabilidad distintas en contextos de contrato distintos. El paper evalúa sobre todas las instancias del benchmark, no sobre cuerpos únicos.

### 1.4 Extracción de función por rango de líneas

**Decisión:** Se usa `source_mapping` del JSON de Slither (cuando está disponible vía validación) y líneas del BugLog para localizar la función en el source. Se extrae el código completo de la función.  
**Justificación:** El paper extrae "function code of vulnerabilities" — retener la función completa (no solo la línea buggy) preserva el contexto sintáctico necesario para que CodeBERT aprenda patrones estructurales.

---

## 2. Dataset SmartBugs Wild (conjunto de TRAIN)

**Fuente:** `datasets/smartbugs_wild.csv` — 47,451 contratos reales de Ethereum (Kaggle: `tranduongminhdai/smartbug-dataset`)  
**Script:** `func_dataset/makeSmartbugs.py --no_filter`  
**Output:** `func_dataset/smartbugs/smartbugs_functions.csv` — 29,908 funciones

### 2.1 Dataset completo sin pre-filtrado MCD

**Decisión:** Se procesan los 47,451 contratos con flag `--no_filter`, sin aplicar Multi-tool Consensus Detection (MCD) previo.  
**Justificación:** El pre-filtrado MCD (T=1.2, µ=0.3) reduciría el dataset a ~5,096 contratos y sesgaría la selección hacia contratos con señal fuerte de múltiples herramientas. El paper usa SmartBugs Wild directamente con Slither como detector único para extraer funciones. La decisión también evita depender de herramientas adicionales (Mythril, SmartCheck) que introducen variabilidad de entorno.

### 2.2 Auto-detección de formato de source code

**Decisión:** Si `src_raw.count('\n') < 5` → aplicar `basic_sol_reformat()` (insertar `\n` después de `{`, `}`, `;`). Si ya tiene múltiples líneas → usar el source directamente.  
**Justificación:** El subset filtrado de SmartBugs (5,096 contratos) tenía el source en una sola línea compacta, requiriendo reformateo para que Slither pueda parsear y mapear líneas. El dataset completo (`smartbugs_wild.csv`) ya tiene source multi-línea; aplicar reformateo sobre código multi-línea desplazaría los números de línea y rompería `extract_function_at_line()`.

### 2.3 Eliminación de comentarios (strip_comments)

**Decisión:** Se eliminan comentarios `//` y `/* */` del source antes de pasarlo a Slither, preservando los saltos de línea (los bloques `/* */` se reemplazan por la misma cantidad de `\n`).  
**Justificación:** El paper menciona explícitamente: *"we removed unrelated characters such as comments and newline characters from the functions to enhance the model's performance."* Preservar los `\n` es crítico para mantener la correspondencia entre números de línea del source y los rangos reportados por Slither en `source_mapping`.

### 2.4 Selección automática de versión de solc

**Decisión:** `detect_solc_version()` parsea el pragma del contrato y selecciona el binario instalado más compatible. Fallback: `solc-0.5.17`.  
**Justificación:** SmartBugs Wild contiene contratos de múltiples versiones de Solidity (0.4.x a 0.8.x). Forzar una versión única (e.g., 0.8.17) hace que la mayoría de contratos con `pragma solidity 0.4.x` fallen en compilación con errores de sintaxis. La selección automática maximiza la tasa de compilación exitosa.

### 2.5 Deduplicación hash cross-contrato

**Decisión:** Se aplica SHA-256 sobre el código normalizado (whitespace colapsado) y se eliminan duplicados globalmente con un `seen_hashes` acumulado.  
**Justificación:** Los contratos reales de Ethereum comparten ampliamente código — librerías como `SafeMath`, patrones ERC20, funciones estándar aparecen en miles de contratos distintos. Sin dedup, la misma función vulnerable (e.g., un `transfer()` con reentrancy idéntico) aparecería cientos de veces, sobreajustando el modelo a esos patrones específicos.

### 2.6 Deduplicación intra-contrato

**Decisión:** Dedup secundaria por `(vuln, address, func_name)` dentro de cada contrato.  
**Justificación:** Un mismo contrato puede reportar el mismo hallazgo en la misma función por múltiples rutas de detección. Evita filas duplicadas antes de la dedup global por hash.

---

## 3. Dataset Slither Audited Smart Contracts (conjunto de TRAIN)

**Fuente:** `datasets/slither-audited/data/raw/contracts*.parquet` — 120,608 contratos (HuggingFace: `mwritescode/slither-audited-smart-contracts`)  
**Script:** `func_dataset/makeSlither.py`  
**Output:** `func_dataset/slither_functions.csv` — 25,180 funciones

### 3.1 Pre-filtrado por resultados pre-computados

**Decisión:** Se usa la columna `results` del dataset (análisis Slither pre-computado sin `source_mapping`) para descartar contratos donde ninguno de los detectores objetivo reportó hallazgos. Reduce 120,608 → 57,884 candidatos (48%).  
**Justificación:** Ejecutar Slither completo (con `--json -` y `source_mapping`) sobre 120,608 contratos sería inviable en tiempo. El pre-filtrado actúa como oráculo: si el Slither pre-computado ya analizó el contrato y no encontró nada relevante, el re-análisis tampoco encontrará. Solo se re-analiza donde hay señal positiva preexistente.

### 3.2 Detector `timestamp` ausente de resultados pre-computados

**Decisión:** Se nota explícitamente que `timestamp` tiene 0 ocurrencias en los 120,608 resultados pre-computados del dataset. No se usa para el pre-filtrado, pero sí se ejecuta en el re-análisis Slither de los candidatos.  
**Justificación:** El dataset de HuggingFace fue generado con una configuración de Slither que no incluyó el detector `timestamp`. Por eso, contratos con Timestamp-Dependency que no tengan otra vulnerabilidad no pasan el pre-filtro y no son analizados. La cobertura de Timestamp-Dependency en este dataset es parcial (solo contratos que también tienen otra vulnerabilidad detectable). La cobertura principal de Timestamp viene de SmartBugs Wild.

### 3.3 Expansión de versiones de solc instaladas

**Decisión:** Se instalaron 32 versiones de solc (0.4.18 a 0.8.17) para cubrir los 95 pragmas distintos presentes en los candidatos.  
**Justificación:** El análisis de pragma distribution mostró que los 8 solc instalados originalmente cubrían ~13% de los contratos (tasa de compilación exitosa). Las versiones más frecuentes (0.8.0: 17,203 contratos; 0.8.4: 7,103; 0.6.0: 3,512) no estaban instaladas. Con 32 versiones, la cobertura de los top pragmas llegó al 100% y la tasa de compilación exitosa subió de ~13% a ~33%.

### 3.4 Mismas decisiones que SmartBugs Wild

Las siguientes decisiones se aplican con la misma justificación que en SmartBugs Wild:
- `strip_comments()` preservando saltos de línea
- `detect_solc_version()` con selección automática por pragma
- Hash dedup cross-contrato por SHA-256 de código normalizado
- Dedup intra-contrato por `(vuln, address, func_name)`
- NO se aplica `basic_sol_reformat()` (el source en los Parquet ya es multi-línea)

### 3.5 Columnas del dataset raw

**Decisión:** Se usan columnas `contracts` (dirección/ID del contrato) y `source_code`. La columna `bytecode` se ignora.  
**Justificación:** Las columnas en el Parquet no siguen la nomenclatura estándar (`address`, `slither`) sino `contracts` y `results`. La columna `bytecode` no es necesaria para análisis estático de source code.

---

## 4. Combinación en Train / Test Split

### 4.1 Separación estricta SolidiFI (test) vs SmartBugs+Slither (train)

**Decisión:** SolidiFI se usa exclusivamente como test set. SmartBugs Wild y Slither Audited forman el training set.  
**Justificación:** El paper usa exactamente esta partición: *"we choose to use the SolidiFI benchmark dataset as our test set."* SolidiFI contiene contratos sintéticos con bugs inyectados — usarlo como training contaminaría el modelo con patrones artificiales de inyección de vulnerabilidades que no aparecen en contratos reales.

### 4.2 Deduplicación cross-dataset en el training set

**Decisión:** Se aplica SHA-256 dedup entre SmartBugs Wild y Slither Audited al combinarlos. Resultado: 55,088 → 52,200 funciones (se eliminan 2,888 duplicados).  
**Justificación:** Ambos datasets toman contratos reales de Ethereum. Es esperable que un mismo contrato aparezca tanto en SmartBugs Wild como en Slither Audited. Sin esta dedup, una función aparecería con peso duplicado en el training, sesgando el modelo hacia esos contratos específicos.

### 4.3 NO se aplica dedup entre test y train

**Decisión:** El test set (SolidiFI) no se filtra para eliminar funciones que coincidan con el training set. El solapamiento es del 2.0% (103 / 5,117 funciones).  
**Justificación:** Las funciones SolidiFI que coinciden con el training son funciones con patrones de vulnerabilidad comunes (e.g., patrones de reentrancy estándar). Eliminarlas sesgaría el test set hacia casos atípicos, inflando artificialmente la dificultad. El paper no menciona este filtrado. El 2.0% de solapamiento es marginal y no compromete la evaluación.

### 4.4 NO se aplica hash dedup al test set (SolidiFI)

**Decisión:** SolidiFI se usa con sus 5,117 funciones completas, sin eliminar duplicados por hash.  
**Justificación:** Ver sección 1.2 y 1.3. Aplicar hash dedup al test set eliminaría el 87% de las muestras (5,117 → 652), convirtiendo la evaluación en algo no representativo del benchmark oficial.

### 4.5 Estructura de labels (one-hot, compatible con BCEWithLogitsLoss)

**Decisión:** Cada fila del CSV tiene una única vulnerabilidad en la columna `vulnerability`. Al cargar para entrenamiento, se convierte a vector one-hot de tamaño 4.  
**Justificación:** El paper usa `BCEWithLogitsLoss`, que aplica binary cross-entropy independientemente en cada una de las 4 salidas del modelo. Un vector one-hot `[1,0,0,0]` es compatible con este criterio — el modelo aprende a activar solo la salida correcta y suprimir las otras 3. Si una función tiene dos vulnerabilidades, aparece como dos filas separadas en el CSV (una por vulnerabilidad detectada), lo que es consistente con el enfoque del paper de extraer funciones *por hallazgo de Slither*.

---

## 5. Resumen de Conteos

| Etapa | Dataset | Entrada | Salida |
|---|---|---|---|
| Extracción | SolidiFI | 200 contratos (50×4 tipos) | 5,117 funciones |
| Extracción | SmartBugs Wild | 47,451 contratos | 29,908 funciones |
| Extracción | Slither Audited | 57,884 candidatos (de 120,608) | 25,180 funciones |
| Dedup cross-dataset | Train (SB + SA) | 55,088 funciones | 52,200 funciones |
| Sin dedup | Test (SolidiFI) | 5,117 funciones | 5,117 funciones |

### Distribución final — Training set (52,200)

| Vulnerabilidad | Cantidad | % |
|---|---|---|
| Re-entrancy | 26,497 | 50.8% |
| Timestamp-Dependency | 18,273 | 35.0% |
| Unhandled-Exceptions | 7,217 | 13.8% |
| tx.origin | 213 | 0.4% |

### Distribución final — Test set (5,117)

| Vulnerabilidad | Cantidad | % |
|---|---|---|
| Re-entrancy | 1,260 | 24.6% |
| Timestamp-Dependency | 1,230 | 24.0% |
| Unhandled-Exceptions | 1,293 | 25.3% |
| tx.origin | 1,334 | 26.1% |

### Origen del training set

| Dataset | Funciones | % |
|---|---|---|
| SmartBugs Wild | 29,908 | 57.3% |
| Slither Audited | 22,292 | 42.7% |

---

## 6. Advertencias y Limitaciones

**Desbalance en tx.origin (train):** Solo 213 muestras de entrenamiento vs 1,334 en test. El paper no aplica balanceo explícito (no menciona `class_weight` ni oversampling). Esto puede resultar en bajo recall para tx.origin. Considerar `pos_weight` en `BCEWithLogitsLoss` como mejora sobre el paper.

**Cobertura parcial de Timestamp-Dependency en Slither Audited:** El detector `timestamp` estaba ausente de los resultados pre-computados del dataset. Los contratos con Timestamp-Dependency como única vulnerabilidad no pasan el pre-filtro. La cobertura de Timestamp en Slither Audited (8,877 funciones) proviene exclusivamente de contratos que también tienen otra vulnerabilidad detectable.

**Dataset expert-audited no replicado:** El paper menciona 1,000 contratos auditados por expertos (1,909 snippets manualmente anotados) como tercera fuente del training set. Esta fuente no está disponible públicamente y no fue reproducida. El training set resultante (52,200) supera el del paper (31,909) por la inclusión del dataset completo de SmartBugs Wild y Slither Audited.

**Tasa de compilación Slither Audited (~33%):** El 67% de los 57,884 candidatos no producen hallazgos, en parte por pragma versions sin cobertura exacta y en parte por source code corrupto en el dataset original. Esto es una limitación del dataset de HuggingFace, no del pipeline.
