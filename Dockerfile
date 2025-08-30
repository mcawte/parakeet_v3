# Slim, production-ready image for Parakeet v3 on Runpod
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# One RUN layer: install deps -> pip install -> quick import check -> purge toolchain -> clean
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        tzdata ffmpeg git-lfs libsndfile1 \
        build-essential g++ gcc make cmake python3-dev pkg-config; \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime; \
    dpkg-reconfigure -f noninteractive tzdata; \
    git lfs install; \
    \
    pip install --no-cache-dir \
        runpod>=1.7.13 \
        "nemo_toolkit[asr]>=2.4.0" \
        einops>=0.6.0 \
        sentencepiece \
        soundfile \
        librosa; \
    \
    # quick import sanity (no GPU required)
    python -c "import torch, einops; import nemo.collections.asr as asr; print('OK:', torch.__version__, einops.__version__)" ; \
    \
    # purge toolchain to slim image
    apt-get purge -y --auto-remove \
        build-essential g++ gcc make cmake python3-dev pkg-config; \
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
