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
# Entorno: Kaggle, GPU Accelerator = T4 x2
# =============================================================================

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
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
# Configuración
# =============================================================================

# Rutas — ajustar según el nombre del dataset subido a Kaggle
TRAIN_CSV = "/kaggle/input/datasets/frmorales/my-dataset/train_functions.csv"
TEST_CSV = "/kaggle/input/datasets/frmorales/my-dataset/test_functions.csv"
OUTPUT_DIR = "/kaggle/working/optimized_codebert"

# Hiperparámetros (Tabla 2, paper)
MAX_LEN = 512
BATCH_SIZE = 16  # por paso real (8 por GPU en T4x2)
GRAD_ACCUM = 8  # batch efectivo = 16 * 8 = 128 (igual al paper)
EPOCHS = 15  # Kaggle T4: ~38 min/época → ~9.5 h total
# Paper usa 60 épocas pero converge antes con LR=1e-3
EARLY_STOP_PAT = 5  # parar si F1_macro no mejora en N épocas consecutivas
# Learning rates diferenciales: el encoder pre-entrenado requiere LR muy bajo
# para no destruir los pesos de CodeBERT (fine-tuning estándar BERT: ~2e-5).
# El paper reporta LR=1e-3 que aplica a la cabeza MLP, no al encoder completo.
LR_ENCODER = 2e-5  # encoder RoBERTa (fine-tuning conservador)
LR_HEAD = 1e-3  # cabeza MLP (igual al paper)
LR_GAMMA = 0.98  # decay exponencial por época (aplica a ambos grupos)
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4  # L2 regularization

# Etiquetas — mismo orden en train y test
VULN_CLASSES = ["Re-entrancy", "Timestamp-Dependency", "Unhandled-Exceptions", "tx.origin"]
NUM_LABELS = len(VULN_CLASSES)
VULN2IDX = {v: i for i, v in enumerate(VULN_CLASSES)}

SEED = 42  # fija la partición train/val para replicabilidad
VAL_SPLIT = 0.2  # 20% de train_functions.csv se reserva para validación interna

# =============================================================================
# Entorno
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

print(f"Dispositivo    : {DEVICE}")
print(f"GPUs detectadas: {NUM_GPUS}")
if NUM_GPUS > 0:
    for i in range(NUM_GPUS):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Dataset
# =============================================================================


class VulnDataset(Dataset):
    """
    Carga un CSV con columnas [function_code, vulnerability] y construye
    tensores one-hot de tamaño NUM_LABELS compatibles con BCEWithLogitsLoss.
    """

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
    CodeBERT-base con cabeza clasificadora de 2 capas MLP.
    Se usa el embedding [CLS] del último encoder como representación del código.

    Arquitectura:
        RobertaEncoder(codebert-base) → [CLS] 768-dim
        → Linear(768, 256) → GELU → Dropout(p)
        → Linear(256, num_labels)   [logits, sin activación]
    """

    def __init__(self, num_labels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        # Gradient checkpointing: recomputa activaciones en backprop en lugar
        # de guardarlas. Reduce memoria ~50% a costa de ~20% más cómputo.
        self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # token [CLS]
        logits = self.classifier(cls)
        return logits  # (batch, num_labels)


# =============================================================================
# Funciones de entrenamiento y evaluación
# =============================================================================


def compute_class_weights(dataset, num_classes, device):
    ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(ds))

    labels = torch.stack([ds.labels[i] for i in indices])  # (N, num_classes)
    total = labels.shape[0]

    cant_clase = labels.sum(dim=0).clamp(min=1)  # evitar div/0
    weights = (total / (num_classes * cant_clase)).clamp(max=10.0)  # evitar explosión

    print("  pos_weight → " + "  ".join(f"{cls}: {w:.2f}" for cls, w in zip(VULN_CLASSES, weights.cpu())))
    return weights.to(device)


def train_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  entrenando", leave=False, ncols=80)):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            logits = model(input_ids, attn_mask)
            # Escalar la loss por grad_accum para que el gradiente sea equivalente
            # a un batch de tamaño BATCH_SIZE * grad_accum
            loss = criterion(logits, labels) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum  # desescalar para el log

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  evaluando", leave=False, ncols=80):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast("cuda"):
                logits = model(input_ids, attn_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu().float())
            all_labels.append(labels.cpu().float())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    preds = (probs > 0.5).astype(float)

    metrics = {
        "loss": total_loss / len(loader),
        "f1_macro": f1_score(all_labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(all_labels, preds, average="micro", zero_division=0),
        "recall_macro": recall_score(all_labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(all_labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(all_labels, preds),
    }
    return metrics, all_labels, preds


def print_per_class_metrics(labels_np, preds_np):
    """Imprime métricas por clase, replicando la Tabla 5 del paper."""
    print(f"\n{'Vulnerabilidad':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 58)
    for i, cls in enumerate(VULN_CLASSES):
        y_true = labels_np[:, i]
        y_pred = preds_np[:, i]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  {cls:<26} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f}")
    print("-" * 58)
    # Macro promedio
    f1_m = f1_score(labels_np, preds_np, average="macro", zero_division=0)
    rec_m = recall_score(labels_np, preds_np, average="macro", zero_division=0)
    prec_m = precision_score(labels_np, preds_np, average="macro", zero_division=0)
    acc_m = accuracy_score(labels_np, preds_np)
    print(f"  {'MACRO':<26} {acc_m:>6.4f} {prec_m:>6.4f} {rec_m:>6.4f} {f1_m:>6.4f}")


def save_checkpoint(model, epoch, metrics, path):
    """Guarda state_dict, unwrapeando DataParallel si es necesario."""
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"epoch": epoch, "metrics": metrics, "state_dict": state}, path)


def load_checkpoint(path, device):
    """Carga un checkpoint y retorna un modelo listo para inferencia."""
    ckpt = torch.load(path, map_location=device)
    model = OptimizedCodeBERT(num_labels=NUM_LABELS, dropout=DROPOUT)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


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
    """
    Genera un dict con toda la información relevante del run para incluir
    en la tesis: hardware, software, hiperparámetros, tiempos y resultados.
    """
    total_s = run_end_ts - run_start_ts
    epoch_times = [r.get("epoch_time_s", 0) for r in history if "epoch_time_s" in r]

    # --- Hardware ---
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpu_info.append(
            {
                "index": i,
                "name": p.name,
                "vram_gb": round(p.total_memory / 1e9, 2),
                "cuda_capability": f"{p.major}.{p.minor}",
                "multi_processors": p.multi_processor_count,
            }
        )

    try:
        import psutil

        ram_total = round(psutil.virtual_memory().total / 1e9, 2)
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
    except ImportError:
        ram_total = cpu_cores = cpu_threads = None

    # --- Software ---
    def pkg_ver(name):
        try:
            import importlib.metadata

            return importlib.metadata.version(name)
        except Exception:
            return "?"

    report = {
        "run_info": {
            "script": "kaggle.py",
            "platform": "Kaggle (GPU T4 x2)",
            "start_utc": datetime.utcfromtimestamp(run_start_ts).isoformat() + "Z",
            "end_utc": datetime.utcfromtimestamp(run_end_ts).isoformat() + "Z",
            "total_time": str(timedelta(seconds=int(total_s))),
            "total_seconds": round(total_s, 1),
        },
        "hardware": {
            "gpus": gpu_info,
            "gpu_count_used": 1,
            "note": "gradient_checkpointing activo; DataParallel desactivado",
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
        },
        "model": {
            "base": "microsoft/codebert-base",
            "architecture": "RobertaModel → [CLS] → Linear(768,256) → GELU → Dropout → Linear(256,4)",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "gradient_checkpointing": True,
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
            "lr_scheduler": f"ExponentialLR(gamma={LR_GAMMA}, aplica a ambos grupos)",
            "optimizer": "Adam",
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "loss_function": "BCEWithLogitsLoss",
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
                (
                    r["epoch"]
                    for r in reversed(history)
                    if r.get("f1_macro") == max(r2.get("f1_macro", 0) for r2 in history)
                ),
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
    return report


# =============================================================================
# Main
# =============================================================================


def main():
    run_start_ts = time.time()

    # -------------------------------------------------------------------------
    # 1. Tokenizador
    # -------------------------------------------------------------------------
    print("\nCargando tokenizador CodeBERT...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # -------------------------------------------------------------------------
    # 2. Datasets y DataLoaders
    # -------------------------------------------------------------------------
    print("Cargando y particionando datasets...")
    full_train_df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=full_train_df["vulnerability"],
    )
    print(
        f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  (split {100 * (1 - VAL_SPLIT):.0f}/{100 * VAL_SPLIT:.0f}, seed={SEED}, estratificado)"
    )
    print("\n  Distribución train:")
    for cls, n in train_df["vulnerability"].value_counts().items():
        print(f"    {cls:<28}: {n:>6,}  ({100 * n / len(train_df):.1f}%)")

    train_ds = VulnDataset(train_df, tokenizer, MAX_LEN)
    val_ds = VulnDataset(val_df, tokenizer, MAX_LEN)
    test_ds = VulnDataset(TEST_CSV, tokenizer, MAX_LEN)
    print(f"\n  Test (held-out): {len(test_ds):,}  ({TEST_CSV})")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"\n  Batch por paso : {BATCH_SIZE} ({BATCH_SIZE // max(NUM_GPUS, 1)} por GPU)")
    print(f"  Grad accum     : {GRAD_ACCUM}  →  batch efectivo: {BATCH_SIZE * GRAD_ACCUM}")

    # -------------------------------------------------------------------------
    # 3. Modelo
    # -------------------------------------------------------------------------
    print("\nInicializando Optimized-CodeBERT...")
    model = OptimizedCodeBERT(num_labels=NUM_LABELS, dropout=DROPOUT)
    model = model.to(DEVICE)

    # gradient_checkpointing es incompatible con DataParallel.
    # Con checkpointing activo, una sola T4 de 16 GB es suficiente para
    # batch=16 + seq=512. Se desactiva DataParallel intencionalmente.
    if NUM_GPUS > 1:
        print(f"  {NUM_GPUS} GPUs disponibles — usando GPU 0 con gradient_checkpointing")
        print("  (DataParallel desactivado: incompatible con gradient_checkpointing)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales   : {total_params:,}")
    print(f"  Parámetros trainables: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # 4. Optimizer, Scheduler, Loss, Scaler
    # -------------------------------------------------------------------------
    optimizer = Adam(
        [
            {"params": model.encoder.parameters(), "lr": LR_ENCODER},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
    class_weights = compute_class_weights(train_ds, 4, DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    scaler = GradScaler("cuda")

    # -------------------------------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------------------------------
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0  # contador para early stopping
    history = []

    print(f"\n{'=' * 70}")
    print(
        f"Entrenamiento: {EPOCHS} épocas máx | LR_encoder={LR_ENCODER} LR_head={LR_HEAD} | batch_efectivo={BATCH_SIZE * GRAD_ACCUM} | early_stop={EARLY_STOP_PAT}"
    )
    print(f"{'=' * 70}")

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE, GRAD_ACCUM)
        torch.cuda.empty_cache()
        val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        torch.cuda.empty_cache()

        scheduler.step()

        row = {"epoch": epoch, "lr": current_lr, "train_loss": train_loss, **val_metrics}
        history.append(row)

        print(
            f"Ep {epoch:>3}/{EPOCHS}  "
            f"lr={current_lr:.2e}  "
            f"loss_tr={train_loss:.4f}  "
            f"loss_val={val_metrics['loss']:.4f}  "
            f"F1_macro={val_metrics['f1_macro']:.4f}  "
            f"Acc={val_metrics['accuracy']:.4f}"
        )

        # Guardar checkpoint por época (sobrescribe el último)
        save_checkpoint(
            model,
            epoch,
            val_metrics,
            os.path.join(OUTPUT_DIR, "checkpoint_last.pt"),
        )

        # Guardar mejor modelo y controlar early stopping
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                model,
                epoch,
                val_metrics,
                os.path.join(OUTPUT_DIR, "best_model.pt"),
            )
            print(f"  *** Mejor modelo guardado (F1_macro={best_f1:.4f}) ***")
        else:
            no_improve += 1
            print(f"  (sin mejora: {no_improve}/{EARLY_STOP_PAT})")
            if no_improve >= EARLY_STOP_PAT:
                print(f"\n  Early stopping: {EARLY_STOP_PAT} épocas sin mejora en F1_macro.")
                break

    # -------------------------------------------------------------------------
    # 6. Evaluación final con el mejor modelo
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Evaluación final sobre TEST SET — mejor modelo (época {best_epoch}, val_F1={best_f1:.4f})")
    print(f"{'=' * 70}")

    best_model = load_checkpoint(os.path.join(OUTPUT_DIR, "best_model.pt"), DEVICE)
    if NUM_GPUS > 1:
        best_model = nn.DataParallel(best_model)

    final_metrics, labels_np, preds_np = evaluate(best_model, test_loader, criterion, DEVICE)

    print("\nMétricas globales:")
    print(f"  F1  macro  : {final_metrics['f1_macro']:.4f}")
    print(f"  F1  micro  : {final_metrics['f1_micro']:.4f}")
    print(f"  Recall mac : {final_metrics['recall_macro']:.4f}")
    print(f"  Precision  : {final_metrics['precision_macro']:.4f}")
    print(f"  Accuracy   : {final_metrics['accuracy']:.4f}")

    print_per_class_metrics(labels_np, preds_np)

    # -------------------------------------------------------------------------
    # 7. Guardar artefactos
    # -------------------------------------------------------------------------
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

    tokenizer.save_pretrained(OUTPUT_DIR)

    # Reporte completo del run
    run_end_ts = time.time()
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
        run_start_ts=run_start_ts,
        run_end_ts=run_end_ts,
        total_params=total_params,
        trainable_params=trainable_params,
    )
    report_path = os.path.join(OUTPUT_DIR, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nArtefactos guardados en: {OUTPUT_DIR}")
    print("  best_model.pt          — pesos del mejor modelo")
    print("  checkpoint_last.pt     — último checkpoint")
    print("  training_history.csv   — curva de entrenamiento")
    print("  run_report.json        — reporte completo del run (hardware, SW, resultados)")


if __name__ == "__main__":
    main()
