FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# --- prevent tzdata from prompting during build ---
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y --no-install-recommends ffmpeg git-lfs libsndfile1 && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface

RUN pip install --no-cache-dir \
      runpod>=1.7.13 \
      "nemo_toolkit[asr]>=2.4.0" \
      soundfile \
      librosa

WORKDIR /app
COPY handler.py /app/handler.py

ENV RUNPOD_SERVERLESS=1
ENV SKIP_MODEL_LOAD=0

# Optional: bake weights to eliminate first-call download
# RUN python -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v3')"

CMD ["python","-u","/app/handler.py"]
