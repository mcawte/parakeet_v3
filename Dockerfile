# Slim, production-ready image for Parakeet v3 on Runpod
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/root/.cache/huggingface

# Single RUN: system deps -> upgrade torch -> pip deps -> sanity check -> purge toolchain -> clean
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        tzdata ffmpeg git-lfs libsndfile1 \
        build-essential g++ gcc make cmake python3-dev pkg-config; \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime; \
    dpkg-reconfigure -f noninteractive tzdata; \
    git lfs install || true; \
    \
    # Upgrade to PyTorch 2.4.x (cu121) so torch.nn.attention is present
    pip uninstall -y torch torchvision torchaudio || true; \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1; \
    \
    # NeMo ASR + extras
    pip install --no-cache-dir \
      huggingface_hub[cli]>=0.23.2 \
      hf_transfer \
      transformers>=4.41 \
      runpod>=1.7.13 \
      "nemo_toolkit[asr]>=2.4.0" \
      einops>=0.6.0 \
      sentencepiece \
      soundfile \
      librosa; \
    \
    # Import sanity (no GPU required)
    python -c "import importlib, torch, einops; import nemo.collections.asr as asr; \
               print('Torch:', torch.__version__); \
               print('Has torch.nn.attention:', importlib.util.find_spec('torch.nn.attention') is not None); \
               print('NeMo ASR import OK')" ; \
    \
    # Purge build toolchain & clean caches to slim image
    apt-get purge -y --auto-remove build-essential g++ gcc make cmake python3-dev pkg-config; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

WORKDIR /app
COPY handler.py /app/handler.py

# Serverless defaults (override at runtime if needed)
ENV RUNPOD_SERVERLESS=1
ENV SKIP_MODEL_LOAD=0
ENV PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3

# Optional: bake weights (adds ~1–2 GB)
# RUN python -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v3')"

CMD ["python","-u","/app/handler.py"]
