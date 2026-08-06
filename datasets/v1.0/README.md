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
- No filtra por contrato principal. Lee todo el contenido de los contratos.
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
  120,608/120,608 (100%)  15,508s
vulnerability
Re-entrancy             10114
Timestamp-Dependency     9556
Unhandled-Exceptions      203
tx.origin                  76

## python makeSmartbugs.py
47,451 contratos cargados, 47,331 con codigo | 16 workers
  47,331/47,331 (100%)  6,316s
vulnerability
Re-entrancy             14455
Timestamp-Dependency    13051
Unhandled-Exceptions      367
tx.origin                  85

## python makeSolidifi.py
Re-entrancy                1,343 funciones
Timestamp-Dependency       1,131 funciones
Unhandled-Exceptions       1,369 funciones
tx.origin                  1,336 funciones
TOTAL                      5,179 funciones

## python makeSplit.py
SmartBugs: 27,958 | Slither: 19,949 | SolidiFI: 5,179
Dedup Slither vs SmartBugs: -12,516 funciones
Dedup Test -> Train:        -0 funciones

TRAIN: 35,391 funciones
vulnerability
Re-entrancy             18192
Timestamp-Dependency    16627
Unhandled-Exceptions      450
tx.origin                 122

TEST: 5,179 funciones
vulnerability
Unhandled-Exceptions    1369
Re-entrancy             1343
tx.origin               1336
Timestamp-Dependency    1131

---