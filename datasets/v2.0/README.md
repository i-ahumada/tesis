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

---

## python makeSlither.py
120,608 contratos cargados, 120,608 con codigo | 16 workers
  120,608/120,608 (100%)  16,078s
vulnerability
Re-entrancy             7855
Timestamp-Dependency    7606
Unhandled-Exceptions     195
tx.origin                 52

## python makeSmartbugs.py
47,451 contratos cargados, 47,331 con codigo | 16 workers
  47,331/47,331 (100%)  6,882s

vulnerability
Re-entrancy             10692
Timestamp-Dependency     9913
Unhandled-Exceptions      331
tx.origin                  70

## python makeSolidifi.py
Re-entrancy                1,343 funciones
Timestamp-Dependency       1,131 funciones
Unhandled-Exceptions       1,369 funciones
tx.origin                  1,336 funciones
TOTAL                      5,179 funciones

## python makeSplit.py
SmartBugs: 21,006 | Slither: 15,708 | SolidiFI: 5,179
Dedup Slither vs SmartBugs: -9,255 funciones
Dedup Test -> Train:        -0 funciones

Validaciones:

TRAIN: 27,459 funciones
vulnerability
Re-entrancy             13892
Timestamp-Dependency    13064
Unhandled-Exceptions      404
tx.origin                  99

TEST: 5,179 funciones
vulnerability
Unhandled-Exceptions    1369
Re-entrancy             1343
tx.origin               1336
Timestamp-Dependency    1131

---