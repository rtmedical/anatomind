# ===== Base com CUDA 12.1 + cuDNN (Ubuntu 22.04) =====
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===== Sistema & Python =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git curl ca-certificates tini build-essential \
 && rm -rf /var/lib/apt/lists/*

# ===== Venv =====
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip/setuptools/wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ===== PyTorch com CUDA 12.1 (compatível com Tesla P40 sm_61) =====
RUN pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121

# ===== Libs do projeto =====
RUN pip install --no-cache-dir \
  transformers==4.46.3 \
  datasets==2.20.0 \
  peft==0.11.1 \
  accelerate==0.34.2 \
  pandas==2.2.2 \
  rapidfuzz==3.9.7 \
  uvicorn==0.30.6 fastapi==0.111.0 \
  sentencepiece==0.2.0 \
  safetensors==0.4.4 \
  einops==0.8.0 \
  hf-transfer==0.1.6 \
  flask
# (Opcional) bitsandbytes para Tesla P40
RUN pip install --no-cache-dir bitsandbytes==0.43.1 --extra-index-url=https://pypi.nvidia.com

# ===== Usuário não-root =====
ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace

# Copiar código da API
COPY --chown=${USERNAME}:${USERNAME} src/ /workspace/

RUN chown -R ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}

# Expor porta da API Flask
EXPOSE 5000

ENTRYPOINT ["/usr/bin/tini", "--"]
# Comando padrão - rodar a API Flask
CMD ["python3", "main.py"]