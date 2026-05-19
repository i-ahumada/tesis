"""
env_info.py
------------------------------------
Captura y anota toda la informacion relevante del entorno de ejecucion
para garantizar la replicabilidad del experimento.

Genera un archivo JSON con:
  - Fecha y hora de ejecucion
  - Sistema operativo y version
  - CPU, RAM disponible
  - Version de Python y arquitectura
  - Versiones de todas las librerias relevantes del pipeline
  - Version de Slither y solc activo
  - Tiempo de procesamiento (si se le pasa el CSV resultante)
  - Hash SHA-256 del CSV de salida (integridad del dataset)

Uso standalone:
  python env_info.py --output env_report.json

Uso desde otro script (para anotar tiempo de procesamiento):
  from env_info import capture_env, save_report
  info = capture_env()
  info["processing_time_s"] = elapsed
  info["output_csv_rows"] = len(df)
  save_report(info, "env_report.json")
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas standalone (cuando se ejecuta directamente: python env_info.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_STANDALONE_OUTPUT = _HERE / "env_report.json"
_STANDALONE_CSV = None  # no hay CSV asociado en modo standalone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list) -> str:
    """Ejecuta un comando y retorna stdout, o '' si falla."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def _pkg_version(pkg: str) -> str:
    """Retorna la version instalada de un paquete pip, o 'not installed'."""
    out = _run([sys.executable, "-m", "pip", "show", pkg])
    for line in out.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "not installed"


def _import_version(module: str, attr: str = "__version__") -> str:
    """Importa un modulo y retorna su __version__, o 'not installed'."""
    try:
        mod = __import__(module)
        return getattr(mod, attr, "unknown")
    except ImportError:
        return "not installed"


def _file_hash(path: str) -> str:
    """SHA-256 del archivo, o '' si no existe."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Captura de entorno
# ---------------------------------------------------------------------------


def capture_env(csv_output_path: str = None) -> dict:
    """
    Captura toda la informacion del entorno de ejecucion.
    Si se pasa csv_output_path, incluye hash y numero de filas del CSV.
    """
    info = {}

    # --- Timestamp ---
    now = datetime.datetime.now()
    info["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    info["timestamp_local"] = now.isoformat()

    # --- Sistema operativo ---
    info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }

    # --- CPU y RAM ---
    try:
        import psutil

        info["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        }
        vm = psutil.virtual_memory()
        info["ram"] = {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
        }
    except ImportError:
        info["cpu"] = "psutil not installed"
        info["ram"] = "psutil not installed"

    # --- Python ---
    info["python"] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "architecture": platform.architecture()[0],
    }

    # --- Librerias del pipeline ---
    LIBS = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("slither-analyzer", None),  # se consulta via pip show
        ("crytic-compile", None),
        ("solc-select", None),
        ("scikit-learn", "sklearn"),
        ("torch", "torch"),
        ("transformers", "transformers"),
    ]

    lib_versions = {}
    for pkg, module in LIBS:
        if module:
            v = _import_version(module)
            if v == "not installed":
                v = _pkg_version(pkg)
        else:
            v = _pkg_version(pkg)
        lib_versions[pkg] = v

    info["libraries"] = lib_versions

    # --- Slither ---
    slither_version = _run(["slither", "--version"])
    info["slither"] = {
        "version": slither_version or "not found",
        "path": _run(["where", "slither"])
        if platform.system() == "Windows"
        else _run(["which", "slither"]),
    }

    # --- solc activo y versiones instaladas (via solc-select) ---
    solc_version = _run(["solc", "--version"])
    solcsel_all = _run(["solc-select", "versions"])
    active_solc = ""
    installed_versions = []
    for line in solcsel_all.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        if "current" in line_s or "(current" in line_s:
            active_solc = line_s
        installed_versions.append(line_s)

    # Paths a binarios de solc-select (usado por makeSmartbugs.py)
    solc_artifacts = os.path.join(os.path.expanduser("~"), ".solc-select", "artifacts")
    solc_bins = {}
    if os.path.isdir(solc_artifacts):
        for d in sorted(os.listdir(solc_artifacts)):
            if d.startswith("solc-"):
                ver = d[5:]
                bin_path = os.path.join(solc_artifacts, d, d)
                solc_bins[ver] = bin_path

    info["solc"] = {
        "active_version": active_solc
        or (solc_version.split("\n")[0] if solc_version else "unknown"),
        "solc_output": solc_version.split("\n")[0] if solc_version else "not found",
        "installed_versions": installed_versions,
        "binary_paths": solc_bins,
        "artifacts_dir": solc_artifacts,
    }

    # --- Variables de entorno relevantes ---
    relevant_env = {}
    for key in (
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "SOLC_VERSION",
        "HOME",
        "APPDATA",
    ):
        val = os.environ.get(key)
        if val:
            # Truncar PATH largo para legibilidad
            relevant_env[key] = val[:300] + "..." if len(val or "") > 300 else val
    info["environment_vars"] = relevant_env

    # --- CSV de salida (integridad) ---
    if csv_output_path and Path(csv_output_path).exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_output_path)
            info["output_csv"] = {
                "path": str(Path(csv_output_path).resolve()),
                "rows": len(df),
                "columns": list(df.columns),
                "sha256": _file_hash(csv_output_path),
                "vuln_counts": df["vulnerability"].value_counts().to_dict()
                if "vulnerability" in df.columns
                else {},
            }
        except Exception as e:
            info["output_csv"] = {"error": str(e)}

    return info


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------


def save_report(info: dict, output_path: str) -> None:
    """Guarda el reporte como JSON indentado."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    print(f"Reporte guardado: {output_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    print("Capturando informacion del entorno...")
    t0 = time.time()
    info = capture_env(csv_output_path=str(_STANDALONE_CSV) if _STANDALONE_CSV else None)
    info["env_capture_time_s"] = round(time.time() - t0, 2)

    save_report(info, str(_STANDALONE_OUTPUT))

    # Vista rapida en consola
    print(f"\nOS         : {info['os']['system']} {info['os']['release']}")
    print(f"Python     : {info['python']['version']}")
    print(f"Slither    : {info['slither']['version']}")
    print(f"solc activo: {info['solc']['active_version']}")
    print(f"pandas     : {info['libraries']['pandas']}")
    if "output_csv" in info and "rows" in info.get("output_csv", {}):
        print(f"CSV filas  : {info['output_csv']['rows']}")
        print(f"CSV sha256 : {info['output_csv']['sha256'][:16]}...")


if __name__ == "__main__":
    main()
