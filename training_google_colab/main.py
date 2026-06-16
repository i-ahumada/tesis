# =============================================================================
# Optimized-CodeBERT para detección de vulnerabilidades en Smart Contracts
# Replicación de: Tang et al., Scientific Reports 2023
#   doi: 10.1038/s41598-023-47219-0
#
# Arquitectura (Tabla 2 del paper):
#   CodeBERT-base → [CLS] (768) → Linear(768,256) → GELU → Dropout(0.1)
#                                → Linear(256,4) → BCEWithLogitsLoss
#
# Hiperparámetros del paper:
#   Epochs=60 | Batch=128 | LR=1e-3 | LR_decay_gamma=0.98 | L2=1e-4
#   Dropout=0.1 | Optimizer=Adam | Loss=BCEWithLogitsLoss
#
# Entorno: Google Colab Pro — versión de entorno de ejecución: 2026.04 (más reciente)
#
# Hardware elegido: GPU A100 (40 GB VRAM)
# ──────────────────────────────────────────────────────────────────────────────
# Por qué A100 y no las demás opciones disponibles en Colab Pro:
#
#   H100 (80 GB VRAM):  Es la opción más potente, pero la disponibilidad en
#                       Colab es baja y consume más "compute units" por hora.
#                       Para este modelo (~125 M params, seq=512) el A100
#                       ya ejecuta las 60 épocas en ~3-5 h. El H100 no
#                       aceleraría el resultado final, solo el tiempo.
#
#   G4 (24 GB VRAM):    Menos VRAM y menor throughput que el A100. Requeriría
#                       batch_size reducido y la ventaja sobre L4 es marginal.
#
#   L4 (24 GB VRAM):    Capaz, pero con 16 GB menos que el A100. Requiere
#                       batch_size=16 y grad_accum=8, igual que el entorno
#                       Kaggle T4x2. No hay razón para tenerla si hay A100.
#
#   T4 (16 GB VRAM):    Ya cubierto por training_kaggle/main.py.
#
#   TPU v6e-1 / v5e-1:  Requieren reescribir todo el pipeline en JAX/XLA
#                       (o usar torch_xla). Incompatible con el código
#                       PyTorch existente sin un refactor significativo.
#
#   CPU:                Inviable. Fine-tuning de CodeBERT llevaría semanas.
#
# El A100 permite:
#   - BF16 nativo (bfloat16): más estable numéricamente que FP16 en A100,
#     no necesita GradScaler para evitar underflow.
#   - batch_size=32 con gradient_checkpointing activo (margen amplio de VRAM).
#   - ~60 épocas en ~3-5 horas.
#   - Acceso a 40 GB VRAM: sin ajustes finos de memoria.
#
# El script detecta automáticamente el tipo de GPU disponible y ajusta
# BATCH_SIZE, GRAD_ACCUM y DTYPE si se asigna una GPU diferente.
# =============================================================================

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Montaje de Google Drive (solo en Colab)
# ---------------------------------------------------------------------------
try:
    from google.colab import drive

    drive.mount("/content/drive")
    IN_COLAB = True
    print("Google Drive montado en /content/drive")
except ImportError:
    IN_COLAB = False
    print("No se detectó entorno Colab — usando rutas locales")

# ---------------------------------------------------------------------------
# Instalación de dependencias faltantes
# Colab 2026.04 incluye PyTorch, numpy y pandas de base.
# Sólo se instalan las que pueden no estar presentes.
# ---------------------------------------------------------------------------
_PKGS = ["transformers>=4.35.0", "scikit-learn>=1.3.0", "tqdm>=4.65.0", "psutil>=5.9.0", "seaborn>=0.12.0"]
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *_PKGS], check=False)

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

# =============================================================================
# Detección de GPU y configuración automática de hiperparámetros de memoria
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_GPU_NAME = ""
_VRAM_GB = 0.0

if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    _GPU_NAME = _props.name
    _VRAM_GB = _props.total_memory / 1e9
    print(f"GPU detectada : {_GPU_NAME}  ({_VRAM_GB:.1f} GB VRAM)")
else:
    print("ADVERTENCIA: No se detectó GPU. El entrenamiento será extremadamente lento.")

# Determinar dtype y batch size según la GPU disponible.
# A100/H100 → BF16 (nativo, más estable). T4/L4/resto → FP16.
_IS_AMPERE_OR_NEWER = any(x in _GPU_NAME for x in ("A100", "H100", "A10", "A30", "A40", "L4", "L40"))

if "H100" in _GPU_NAME and _VRAM_GB >= 70:
    # H100 80 GB: batch más grande, grad_accum mínimo
    _BATCH_SIZE = 64
    _GRAD_ACCUM = 2
elif "A100" in _GPU_NAME or ("L4" in _GPU_NAME and _VRAM_GB >= 22):
    # A100 40 GB / L40 48 GB: configuración objetivo
    _BATCH_SIZE = 32
    _GRAD_ACCUM = 4
elif "L4" in _GPU_NAME or ("V100" in _GPU_NAME and _VRAM_GB >= 30):
    # L4 24 GB / V100 32 GB
    _BATCH_SIZE = 16
    _GRAD_ACCUM = 8
else:
    # T4 16 GB o desconocida: configuración conservadora
    _BATCH_SIZE = 8
    _GRAD_ACCUM = 16

DTYPE = torch.bfloat16 if _IS_AMPERE_OR_NEWER else torch.float16
USE_SCALER = DTYPE == torch.float16  # BF16 no necesita GradScaler

print(f"Configuración : BATCH={_BATCH_SIZE}  GRAD_ACCUM={_GRAD_ACCUM}  "
      f"batch_efectivo={_BATCH_SIZE * _GRAD_ACCUM}  dtype={DTYPE}")
print(f"GradScaler    : {'activado (FP16)' if USE_SCALER else 'desactivado (BF16)'}")

# =============================================================================
# Rutas — ajustar DRIVE_ROOT si la estructura en Drive es diferente
# =============================================================================

DRIVE_ROOT = "/content/drive/MyDrive/tesis"

TRAIN_CSV = os.getenv("TRAIN_CSV", f"{DRIVE_ROOT}/data/train_functions.csv")
TEST_CSV = os.getenv("TEST_CSV", f"{DRIVE_ROOT}/data/test_functions.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"{DRIVE_ROOT}/runs/optimized_codebert_colab")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Hiperparámetros (Tabla 2 del paper)
# =============================================================================

MAX_LEN = 512
BATCH_SIZE = _BATCH_SIZE
GRAD_ACCUM = _GRAD_ACCUM
EPOCHS = 60          # exacto al paper; el A100 lo completa en ~3-5 h
EARLY_STOP_PAT = 10  # tolerante para convergencia completa
LR_ENCODER = 2e-5    # fine-tuning conservador del encoder pre-entrenado
LR_HEAD = 1e-3       # cabeza MLP (igual al paper)
LR_GAMMA = 0.98      # decay exponencial por época (ambos grupos de parámetros)
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4  # L2 regularization
NUM_WORKERS = 2      # Colab tiene CPUs limitadas; 2 es seguro

VULN_CLASSES = ["Re-entrancy", "Timestamp-Dependency", "Unhandled-Exceptions", "tx.origin"]
NUM_LABELS = len(VULN_CLASSES)
VULN2IDX = {v: i for i, v in enumerate(VULN_CLASSES)}

SEED = 42      # fija la partición train/val para replicabilidad entre entornos
VAL_SPLIT = 0.2

# =============================================================================
# Dataset
# =============================================================================


class VulnDataset(Dataset):
    def __init__(self, data, tokenizer, max_len: int):
        df = pd.read_csv(data) if isinstance(data, str) else data.reset_index(drop=True)
        self.codes = df["function_code"].fillna("").astype(str).tolist()
        self.labels = []
        for vuln in df["vulnerability"]:
            label = torch.zeros(NUM_LABELS)
            if vuln in VULN2IDX:
                label[VULN2IDX[vuln]] = 1.0
            self.labels.append(label)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


# =============================================================================
# Modelo: Optimized-CodeBERT (Fig. 4 + Tabla 2 del paper)
# =============================================================================


class OptimizedCodeBERT(nn.Module):
    """
    CodeBERT-base + cabeza clasificadora MLP de 2 capas.

    Arquitectura:
        RobertaEncoder(codebert-base) → [CLS] 768-dim
        → Linear(768, 256) → GELU → Dropout(p)
        → Linear(256, num_labels)   [logits sin activación]
    """

    def __init__(self, num_labels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", revision="refs/pr/9")
        # Gradient checkpointing: recomputa activaciones en backprop en lugar
        # de almacenarlas. En A100 no es estrictamente necesario por VRAM, pero
        # reduce el pico de memoria y es consistente con los otros entornos.
        self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


# =============================================================================
# Entrenamiento y evaluación
# =============================================================================


def compute_class_weights(dataset, num_classes, device):
    ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(ds))
    labels = torch.stack([ds.labels[i] for i in indices])
    total = labels.shape[0]
    cant_clase = labels.sum(dim=0).clamp(min=1)
    weights = (total / (num_classes * cant_clase)).clamp(max=10.0)
    print("  pos_weight → " + "  ".join(f"{cls}: {w:.2f}" for cls, w in zip(VULN_CLASSES, weights.cpu())))
    return weights.to(device)


def train_epoch(model, loader, optimizer, criterion, scaler, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False, ncols=90)):
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast("cuda", dtype=DTYPE):
            loss = criterion(model(input_ids, attn_mask), labels) / grad_accum

        if USE_SCALER:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if USE_SCALER:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval ", leave=False, ncols=90):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with autocast("cuda", dtype=DTYPE):
                logits = model(input_ids, attn_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu().float())
            all_labels.append(labels.cpu().float())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs > 0.5).astype(float)

    return (
        {
            "loss": total_loss / len(loader),
            "f1_macro": f1_score(all_labels, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(all_labels, preds, average="micro", zero_division=0),
            "recall_macro": recall_score(all_labels, preds, average="macro", zero_division=0),
            "precision_macro": precision_score(all_labels, preds, average="macro", zero_division=0),
            "accuracy": accuracy_score(all_labels, preds),
        },
        all_labels,
        preds,
        probs,
    )


def print_per_class(labels_np, preds_np):
    print(f"\n  {'Vulnerabilidad':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("  " + "-" * 56)
    for i, cls in enumerate(VULN_CLASSES):
        y_t, y_p = labels_np[:, i], preds_np[:, i]
        print(
            f"  {cls:<28} "
            f"{accuracy_score(y_t, y_p):>6.4f} "
            f"{precision_score(y_t, y_p, zero_division=0):>6.4f} "
            f"{recall_score(y_t, y_p, zero_division=0):>6.4f} "
            f"{f1_score(y_t, y_p, zero_division=0):>6.4f}"
        )
    print("  " + "-" * 56)
    print(
        f"  {'MACRO':<28} "
        f"{accuracy_score(labels_np, preds_np):>6.4f} "
        f"{precision_score(labels_np, preds_np, average='macro', zero_division=0):>6.4f} "
        f"{recall_score(labels_np, preds_np, average='macro', zero_division=0):>6.4f} "
        f"{f1_score(labels_np, preds_np, average='macro', zero_division=0):>6.4f}"
    )


def save_checkpoint(model, epoch, metrics, path):
    torch.save({"epoch": epoch, "metrics": metrics, "state_dict": model.state_dict()}, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = OptimizedCodeBERT(num_labels=NUM_LABELS, dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model


# =============================================================================
# Reporte de ejecución (para tesis)
# =============================================================================


def _run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def build_run_report(
    history,
    final_metrics,
    labels_np,
    preds_np,
    train_size,
    val_size,
    test_size,
    train_dist,
    run_start_ts,
    run_end_ts,
    total_params=None,
    trainable_params=None,
):
    total_s = run_end_ts - run_start_ts
    epoch_times = [r.get("epoch_time_s", 0) for r in history if "epoch_time_s" in r]

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpu_info.append({
            "index": i,
            "name": p.name,
            "vram_gb": round(p.total_memory / 1e9, 2),
            "cuda_capability": f"{p.major}.{p.minor}",
            "multi_processors": p.multi_processor_count,
        })

    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total = round(vm.total / 1e9, 2)
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
    except ImportError:
        ram_total = cpu_cores = cpu_threads = None

    def pkg_ver(name):
        try:
            import importlib.metadata
            return importlib.metadata.version(name)
        except Exception:
            return "?"

    return {
        "run_info": {
            "script": "main.py",
            "platform": "Google Colab Pro — entorno 2026.04 (más reciente)",
            "start_utc": datetime.utcfromtimestamp(run_start_ts).isoformat() + "Z",
            "end_utc": datetime.utcfromtimestamp(run_end_ts).isoformat() + "Z",
            "total_time": str(timedelta(seconds=int(total_s))),
            "total_seconds": round(total_s, 1),
        },
        "hardware": {
            "gpus": gpu_info,
            "gpu_target": "A100 (40 GB VRAM)",
            "gpu_detected": _GPU_NAME,
            "dtype": str(DTYPE),
            "grad_scaler": USE_SCALER,
            "gradient_checkpointing": True,
            "cpu_physical_cores": cpu_cores,
            "cpu_logical_threads": cpu_threads,
            "ram_total_gb": ram_total,
            "os": f"{platform.system()} {platform.release()}",
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": str(torch.backends.cudnn.version()),
            "transformers": pkg_ver("transformers"),
            "numpy": pkg_ver("numpy"),
            "pandas": pkg_ver("pandas"),
            "scikit_learn": pkg_ver("scikit-learn"),
            "colab_env": "2026.04",
        },
        "model": {
            "base": "microsoft/codebert-base",
            "revision": "refs/pr/9",
            "architecture": "RobertaModel → [CLS] → Linear(768,256) → GELU → Dropout → Linear(256,4)",
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "hyperparameters": {
            "max_len": MAX_LEN,
            "batch_size_per_step": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "epochs_max": EPOCHS,
            "early_stop_patience": EARLY_STOP_PAT,
            "lr_encoder": LR_ENCODER,
            "lr_head": LR_HEAD,
            "lr_scheduler": f"ExponentialLR(gamma={LR_GAMMA})",
            "optimizer": "Adam",
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "loss_function": "BCEWithLogitsLoss + pos_weight",
        },
        "dataset": {
            "train_csv": TRAIN_CSV,
            "test_csv": TEST_CSV,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "val_split": VAL_SPLIT,
            "seed": SEED,
            "vulnerability_classes": VULN_CLASSES,
            "train_distribution": train_dist,
        },
        "training_summary": {
            "epochs_run": len(history),
            "best_epoch": next(
                (r["epoch"] for r in reversed(history)
                 if r.get("f1_macro") == max(r2.get("f1_macro", 0) for r2 in history)),
                None,
            ),
            "avg_epoch_time_s": round(sum(epoch_times) / len(epoch_times), 1) if epoch_times else None,
            "avg_epoch_time": str(timedelta(seconds=int(sum(epoch_times) / len(epoch_times)))) if epoch_times else None,
            "loss_train_final": round(history[-1]["train_loss"], 6) if history else None,
            "loss_val_final": round(history[-1]["loss"], 6) if history else None,
        },
        "results": {
            "f1_macro": round(final_metrics["f1_macro"], 6),
            "f1_micro": round(final_metrics["f1_micro"], 6),
            "recall_macro": round(final_metrics["recall_macro"], 6),
            "precision_macro": round(final_metrics["precision_macro"], 6),
            "accuracy": round(final_metrics["accuracy"], 6),
            "per_class": {
                cls: {
                    "f1": round(f1_score(labels_np[:, i], preds_np[:, i], zero_division=0), 6),
                    "recall": round(recall_score(labels_np[:, i], preds_np[:, i], zero_division=0), 6),
                    "precision": round(precision_score(labels_np[:, i], preds_np[:, i], zero_division=0), 6),
                    "accuracy": round(float((labels_np[:, i] == preds_np[:, i]).mean()), 6),
                    "support": int(labels_np[:, i].sum()),
                }
                for i, cls in enumerate(VULN_CLASSES)
            },
        },
        "paper_reference": {
            "title": "Deep learning-based solution for smart contract vulnerabilities detection",
            "authors": "Tang et al.",
            "journal": "Scientific Reports",
            "year": 2023,
            "doi": "10.1038/s41598-023-47219-0",
            "reported_f1": 0.9353,
        },
    }


def generate_report(
    history,
    final_metrics,
    labels_np,
    preds_np,
    probs_np,
    train_dist,
    test_csv,
    tokenizer,
    run_name="optimized_codebert_colab",
    hyperparams=None,
    save_dir=OUTPUT_DIR,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(save_dir, f"{run_name}_{timestamp}")

    test_df = pd.read_csv(test_csv)
    codes = test_df["function_code"].fillna("").astype(str).tolist()

    true_class = np.argmax(labels_np, axis=1)
    pred_class = np.argmax(probs_np, axis=1)

    report_dict = classification_report(
        labels_np, preds_np, target_names=VULN_CLASSES, zero_division=0, output_dict=True
    )

    lines = []
    lines.append("=" * 80)
    lines.append(f"REPORTE DE ENTRENAMIENTO: {run_name}")
    lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Entorno: Google Colab Pro 2026.04 — GPU: {_GPU_NAME} ({_VRAM_GB:.1f} GB)")
    lines.append("=" * 80)

    lines.append("\n--- HIPERPARÁMETROS ---")
    if hyperparams:
        for k, v in hyperparams.items():
            lines.append(f"  {k}: {v}")

    lines.append("\n--- MÉTRICAS GENERALES ---")
    lines.append(f"  F1 Macro  : {final_metrics['f1_macro']:.4f}")
    lines.append(f"  F1 Micro  : {final_metrics['f1_micro']:.4f}")
    lines.append(f"  Recall    : {final_metrics['recall_macro']:.4f}")
    lines.append(f"  Precision : {final_metrics['precision_macro']:.4f}")
    lines.append(f"  Accuracy  : {final_metrics['accuracy']:.4f}")

    lines.append("\n--- MÉTRICAS POR CLASE ---")
    lines.append(f"  {'Clase':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("  " + "-" * 68)
    for cls in VULN_CLASSES:
        if cls in report_dict:
            r = report_dict[cls]
            lines.append(
                f"  {cls:<28} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1-score']:>10.4f} {int(r['support']):>10d}"
            )

    lines.append("\n--- DISTRIBUCIÓN TRAIN ---")
    for cls in VULN_CLASSES:
        lines.append(f"  {cls:<28} {train_dist.get(cls, 0):>8,}")

    lines.append("\n--- DISTRIBUCIÓN TEST ---")
    test_dist = test_df["vulnerability"].value_counts()
    for cls in VULN_CLASSES:
        lines.append(f"  {cls:<28} {test_dist.get(cls, 0):>8,}")

    lines.append("\n--- HISTORIAL DE ENTRENAMIENTO ---")
    lines.append(f"  {'Época':>5} {'LR':>10} {'Loss_tr':>10} {'Loss_val':>10} {'F1':>8} {'Acc':>8} {'Tiempo':>10}")
    lines.append("  " + "-" * 75)
    for r in history:
        t_str = str(timedelta(seconds=int(r["epoch_time_s"]))) if "epoch_time_s" in r else "-"
        lines.append(
            f"  {r['epoch']:>5} {r['lr']:>10.2e} {r['train_loss']:>10.4f} "
            f"{r['loss']:>10.4f} {r['f1_macro']:>8.4f} {r['accuracy']:>8.4f} {t_str:>10}"
        )

    lines.append("\n--- MATRIZ DE CONFUSIÓN (argmax de probabilidades) ---")
    cm = confusion_matrix(true_class, pred_class, labels=list(range(NUM_LABELS)))
    lines.append(f"  {'':>22}" + "".join(f"{c[:10]:>13}" for c in VULN_CLASSES))
    for i, cls in enumerate(VULN_CLASSES):
        lines.append(f"  {cls:>22}" + "".join(f"{cm[i][j]:>13}" for j in range(NUM_LABELS)))

    n = len(true_class)
    correct = sum(true_class[i] == pred_class[i] for i in range(n))
    lines.append(f"\nResumen predicciones: {correct}/{n} correctas ({100 * correct / n:.1f}%)")

    report_path = f"{prefix}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = f"{prefix}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "timestamp": timestamp,
                "hyperparams": hyperparams or {},
                "metrics": {
                    "f1_macro": round(final_metrics["f1_macro"], 6),
                    "f1_micro": round(final_metrics["f1_micro"], 6),
                    "recall": round(final_metrics["recall_macro"], 6),
                    "precision": round(final_metrics["precision_macro"], 6),
                    "accuracy": round(final_metrics["accuracy"], 6),
                },
                "per_class": {cls: report_dict.get(cls, {}) for cls in VULN_CLASSES},
                "confusion_matrix": cm.tolist(),
                "train_distribution": train_dist,
                "test_distribution": test_dist.to_dict(),
            },
            f, indent=2, ensure_ascii=False, default=str,
        )

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_path = f"{prefix}_confusion_matrix.png"
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=VULN_CLASSES, yticklabels=VULN_CLASSES, ax=axes[0])
    axes[0].set_xlabel("Predicho"); axes[0].set_ylabel("Real")
    axes[0].set_title("Matriz de Confusión (absoluta)")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=VULN_CLASSES, yticklabels=VULN_CLASSES, ax=axes[1])
    axes[1].set_xlabel("Predicho"); axes[1].set_ylabel("Real")
    axes[1].set_title("Matriz de Confusión (normalizada por fila)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle(run_name, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nReporte guardado:")
    print(f"  Texto   : {report_path}")
    print(f"  Métricas: {json_path}")
    print(f"  Matriz  : {cm_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    run_start = time.time()

    print("=" * 70)
    print("Optimized-CodeBERT — Google Colab Pro 2026.04")
    print(f"GPU     : {_GPU_NAME} ({_VRAM_GB:.1f} GB)  |  dtype: {DTYPE}")
    print(f"Inicio  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TRAIN   : {TRAIN_CSV}")
    print(f"TEST    : {TEST_CSV}")
    print(f"OUTPUT  : {OUTPUT_DIR}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Tokenizador y datasets
    # -------------------------------------------------------------------------
    print("\nCargando tokenizador CodeBERT...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print("Cargando y particionando datasets...")
    full_train_df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=full_train_df["vulnerability"],
    )
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  "
          f"(split {100*(1-VAL_SPLIT):.0f}/{100*VAL_SPLIT:.0f}, seed={SEED}, estratificado)")
    print("\n  Distribución train:")
    for cls, n in train_df["vulnerability"].value_counts().items():
        print(f"    {cls:<28}: {n:>6,}  ({100 * n / len(train_df):.1f}%)")

    train_ds = VulnDataset(train_df, tokenizer, MAX_LEN)
    val_ds = VulnDataset(val_df, tokenizer, MAX_LEN)
    test_ds = VulnDataset(TEST_CSV, tokenizer, MAX_LEN)
    print(f"\n  Test (held-out): {len(test_ds):,}  ({TEST_CSV})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n  Batch/paso     : {BATCH_SIZE}")
    print(f"  Grad accum     : {GRAD_ACCUM}  →  batch efectivo: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  DataLoader workers: {NUM_WORKERS}")

    # -------------------------------------------------------------------------
    # Modelo
    # -------------------------------------------------------------------------
    print("\nInicializando Optimized-CodeBERT...")
    model = OptimizedCodeBERT(num_labels=NUM_LABELS, dropout=DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales   : {total_params:,}")
    print(f"  Parámetros trainables: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # Optimizer, scheduler, loss, scaler
    # -------------------------------------------------------------------------
    optimizer = Adam(
        [
            {"params": model.encoder.parameters(), "lr": LR_ENCODER},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
    class_weights = compute_class_weights(train_ds, NUM_LABELS, DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    scaler = GradScaler("cuda", enabled=USE_SCALER)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    history = []

    print(f"\n{'=' * 70}")
    print(f"Entrenamiento: {EPOCHS} épocas máx | LR_enc={LR_ENCODER} LR_head={LR_HEAD} "
          f"| decay={LR_GAMMA}/época | early_stop={EARLY_STOP_PAT}")
    print(f"{'=' * 70}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, GRAD_ACCUM)
        torch.cuda.empty_cache()
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion)
        torch.cuda.empty_cache()

        scheduler.step()

        elapsed = time.time() - t0
        row = {"epoch": epoch, "lr": current_lr, "epoch_time_s": elapsed, "train_loss": train_loss, **val_metrics}
        history.append(row)

        total_elapsed = timedelta(seconds=int(time.time() - run_start))
        avg_ep = sum(r["epoch_time_s"] for r in history) / len(history)
        eta = timedelta(seconds=int(avg_ep * (EPOCHS - epoch)))

        print(
            f"Ep {epoch:>3}/{EPOCHS}  lr={current_lr:.2e}  "
            f"loss_tr={train_loss:.4f}  loss_val={val_metrics['loss']:.4f}  "
            f"F1={val_metrics['f1_macro']:.4f}  Acc={val_metrics['accuracy']:.4f}  "
            f"[{elapsed/60:.1f}min/ep | total={total_elapsed} | ETA={eta}]"
        )

        # Guardar historial parcial tras cada época (persistente en Drive)
        pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
        save_checkpoint(model, epoch, val_metrics, os.path.join(OUTPUT_DIR, "checkpoint_last.pt"))

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, epoch, val_metrics, os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"  *** Mejor modelo guardado (F1_macro={best_f1:.4f}) ***")
        else:
            no_improve += 1
            print(f"  (sin mejora: {no_improve}/{EARLY_STOP_PAT})")
            if no_improve >= EARLY_STOP_PAT:
                print(f"\nEarly stopping en época {epoch}.")
                break

    # -------------------------------------------------------------------------
    # Evaluación final sobre test set
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Evaluación final sobre TEST SET — mejor modelo (época {best_epoch}, val_F1={best_f1:.4f})")
    print(f"{'=' * 70}")

    best_model = load_checkpoint(os.path.join(OUTPUT_DIR, "best_model.pt"))
    final_metrics, labels_np, preds_np, probs_np = evaluate(best_model, test_loader, criterion)

    print(f"\n  F1  macro  : {final_metrics['f1_macro']:.4f}")
    print(f"  F1  micro  : {final_metrics['f1_micro']:.4f}")
    print(f"  Recall     : {final_metrics['recall_macro']:.4f}")
    print(f"  Precision  : {final_metrics['precision_macro']:.4f}")
    print(f"  Accuracy   : {final_metrics['accuracy']:.4f}")
    print_per_class(labels_np, preds_np)

    # -------------------------------------------------------------------------
    # Artefactos
    # -------------------------------------------------------------------------
    run_end = time.time()
    total_time = timedelta(seconds=int(run_end - run_start))
    print(f"\nTiempo total: {total_time}")

    tokenizer.save_pretrained(OUTPUT_DIR)

    train_dist = train_df["vulnerability"].value_counts().to_dict()

    report = build_run_report(
        history=history,
        final_metrics=final_metrics,
        labels_np=labels_np,
        preds_np=preds_np,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        train_dist=train_dist,
        run_start_ts=run_start,
        run_end_ts=run_end,
        total_params=total_params,
        trainable_params=trainable_params,
    )
    with open(os.path.join(OUTPUT_DIR, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    generate_report(
        history=history,
        final_metrics=final_metrics,
        labels_np=labels_np,
        preds_np=preds_np,
        probs_np=probs_np,
        train_dist=train_dist,
        test_csv=TEST_CSV,
        tokenizer=tokenizer,
        run_name="optimized_codebert_colab",
        hyperparams={
            "model": "microsoft/codebert-base (revision=refs/pr/9)",
            "colab_env": "2026.04",
            "gpu": _GPU_NAME,
            "dtype": str(DTYPE),
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "epochs_max": EPOCHS,
            "early_stop": EARLY_STOP_PAT,
            "lr_encoder": LR_ENCODER,
            "lr_head": LR_HEAD,
            "lr_gamma": LR_GAMMA,
            "optimizer": "Adam",
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "loss": "BCEWithLogitsLoss + pos_weight",
            "gradient_checkpointing": True,
        },
        save_dir=OUTPUT_DIR,
    )

    print(f"\nArtefactos guardados en: {OUTPUT_DIR}/")
    print("  best_model.pt               — mejor modelo")
    print("  checkpoint_last.pt          — último checkpoint")
    print("  training_history.csv        — curva de entrenamiento por época")
    print("  run_report.json             — reporte completo (hardware, SW, resultados)")
    print("  *_report.txt                — reporte detallado con historial completo")
    print("  *_metrics.json              — métricas por clase")
    print("  *_confusion_matrix.png      — heatmaps absoluto y normalizado")


if __name__ == "__main__":
    main()
