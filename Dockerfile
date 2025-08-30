# Stable, public base with CUDA 12.1 + cuDNN8 + PyTorch preinstalled
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git-lfs libsndfile1 && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Speed up HF downloads (optional)
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface

# Python deps — torch is already in the base image
RUN pip install --no-cache-dir \
      runpod>=1.7.13 \
      "nemo_toolkit[asr]>=2.4.0" \
      soundfile \
      librosa

# App
WORKDIR /app
COPY handler.py /app/handler.py

# Serverless on Runpod; flip off locally
ENV RUNPOD_SERVERLESS=1
# Load the model in prod; you can override with -e SKIP_MODEL_LOAD=1
ENV SKIP_MODEL_LOAD=0

# (Optional) Bake weights into the image to eliminate first-call download
# RUN python -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v3')"

CMD ["python","-u","/app/handler.py"]
