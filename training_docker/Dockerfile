# Base: PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9 (runtime — sin herramientas de compilación)
# Documentación del entorno exacto usado para entrenar Optimized-CodeBERT (tesis)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Dependencias Python (PyTorch y CUDA ya vienen en la imagen base)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# -u: salida sin buffer → los logs aparecen en tiempo real en `docker logs`
CMD ["python", "-u", "train.py"]
