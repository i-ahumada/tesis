# Training:
- Usa Smartbugs y Slither Audited Dataset
- Normalización ANTES de ejecutar Slither: comentarios eliminados y código
  re-indentado con jsbeautifier.
- Procesa ambos completamente usando Slither buscando los siguientes detectores:
DETECTORS_BY_VULN = {
    "Re-entrancy": ["reentrancy-eth", "reentrancy-no-eth", "reentrancy-unlimited-gas", "reentrancy-benign"],
    "Timestamp-Dependency": ["timestamp"],
    "Unhandled-Exceptions": ["unchecked-lowlevel", "unchecked-send"],
    "tx.origin": ["tx-origin"],
}
- Filtro de contrato principal: se conservan los hallazgos cuya función pertenece
  al contrato principal o a su cadena de herencia — el código escrito por el
  programador. El principal se identifica como el `contract` con la clausura de
  herencia más grande (más robusto que "el último declarado", que a veces es un
  helper). Se descartan libraries y contratos auxiliares no heredados (SafeMath,
  interfaces sueltas, etc.) pegados en el mismo source_code.
- Deduplicación: (contrato, función, vulnerabilidad) y hash SHA-256 global del código.

# Test:
- Usa Solidifi
- Las funciones vulnerables se identifican por NOMBRE: SolidiFI inyecta funciones
  con sufijos característicos por categoría (_re_entN, _tmstmpN, _txoriginN,
  _unchkN/_uncheckN).
- Solo se extraen funciones: de las 5.434 entradas del BugLog, 5.179 son funciones
  inyectadas; las 255 restantes son declaraciones de variables sueltas y se omiten.
- Normalización: se eliminan comentarios y se re-indenta con jsbeautifier.
- Sin deduplicación por hash.

# Split
- Test: SolidiFI completo, sin dedup interno (el mismo template inyectado en
  varios contratos representa casos de test válidos).
- Train: SmartBugs-Wild + Slither-Audited, con dos deduplicaciones por hash
  SHA-256 del código con espacios colapsados:
    1. Entre datasets: se conserva SmartBugs y se eliminan de Slither las
       funciones cuyo hash ya está en SmartBugs.
    2. Test → Train: se elimina de train toda función cuyo hash aparece en
       test, para evitar data leakage.
- Corre las verificaciones:
    1. Que el snippet inicie con la palabra "function", "constructor", "fallback", "receive", "modifier"
    2. Que el snippet tenga llaves balanceadas
    3. Si un snippet tiene "tx.origin" y no está catalogado con esa vulnerabilidad se elimina
    4. Si un snippet tiene "block.timestamp" o "now" y no está catalogado con esa vulnerabilidad se elimina
    5. Si un snippet tiene mas de 510 tokens se elimina
- Balancea el dataset eliminando snippets de las vulnerabilidades declaradas hasta alcanzar la cantidad deseada. Al hacerlo toma en cuenta los detectores de Slither para no perder casos de distintos detectores.
DROP_VULN_CAPS = {
    "Re-entrancy": 3500,
    "Timestamp-Dependency": 3500,
}
---

## python makeSlither.py
120,608 contratos cargados, 120,608 con codigo | 16 workers
  120,608/120,608 (100%)  14,198s

vulnerability
Re-entrancy             7812
Timestamp-Dependency    7650
Unhandled-Exceptions     198
tx.origin                 50

## python makeSmartbugs.py
47,451 contratos cargados, 47,331 con codigo | 16 workers
  47,331/47,331 (100%)  6,500s

vulnerability
Re-entrancy             10628
Timestamp-Dependency     9991
Unhandled-Exceptions      319
tx.origin                  69

## python makeSolidifi.py
Re-entrancy                1,343 funciones
Timestamp-Dependency       1,131 funciones
Unhandled-Exceptions       1,369 funciones
tx.origin                  1,336 funciones
TOTAL                      5,179 funciones

## python makeSplit.py
SmartBugs: 21,007 | Slither: 15,710 | SolidiFI: 5,179
Dedup Slither vs SmartBugs: -9,251 funciones
Dedup Test -> Train:        -0 funciones

Cargando tokenizador de CodeBERT...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Validaciones:
[TRAIN] 'tx.origin' en etiqueta distinta a tx.origin: -160
[TRAIN] uso de timestamp en etiqueta distinta a Timestamp-Dependency: -2,723
Token indices sequence length is longer than the specified maximum sequence length for this model (739 > 512). Running this sequence through the model will result in indexing errors
[TRAIN] más de 510 tokens: -3,442

Balanceo:
[TRAIN] Re-entrancy / reentrancy-benign: -2,057 (de 3,256)
[TRAIN] Re-entrancy / reentrancy-eth: -175 (de 277)
[TRAIN] Re-entrancy / reentrancy-no-eth: -1,416 (de 2,241)
[TRAIN] Re-entrancy / reentrancy-unlimited-gas: -2,355 (de 3,729)
[TRAIN] Re-entrancy total: -6,003 (cap 3,500)
[TRAIN] Timestamp-Dependency / timestamp: -7,755 (de 11,255)
[TRAIN] Timestamp-Dependency total: -7,755 (cap 3,500)

TRAIN: 7,383 funciones
vulnerability
Re-entrancy             3500
Timestamp-Dependency    3500
Unhandled-Exceptions     312
tx.origin                 71

TEST: 5,179 funciones
vulnerability
Unhandled-Exceptions    1369
Re-entrancy             1343
tx.origin               1336
Timestamp-Dependency    1131
---