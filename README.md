## Overview

This is a Runpod serverless worker for audio transcription using NVIDIA's Parakeet ASR model. The service accepts audio files (URLs or local paths) and returns transcribed text with optional timestamps.

## Architecture

### Core Components

- **handler.py**: Main serverless worker implementation
  - `handler()`: Entry point that processes transcription requests
  - `transcribe_batched()`: Smart batching strategy for efficient GPU utilization
  - `_fetch_to_wav()`: Converts any audio format to 16kHz mono WAV using ffmpeg
  - `_maybe_set_local_attention()`: Dynamically switches attention mechanism based on audio length
  - `_bucketize_by_duration()`: Separates short vs long audio files
  - `_make_batches()`: Packs short audio files into efficient batches

### Batching Strategy

The service implements a hybrid processing approach:

- **Short audio** (≤10 min by default): Batched together for parallel GPU processing
- **Long audio** (>10 min): Processed sequentially to manage memory
- Batch constraints: Max 16 items or 20 minutes total duration per batch
- Results are returned in the original input order

### Model Loading Strategy

The service uses conditional model loading based on environment:

- Production (`SKIP_MODEL_LOAD=0`): Loads NVIDIA Parakeet TDT model via NeMo
- Development (`SKIP_MODEL_LOAD=1`): Dry-run mode without loading heavy dependencies

### Request/Response Flow

1. Accepts inputs via Runpod event format:
   ```json
   {"input": {"timestamps": bool, "inputs": [{"source": "audio_url"}]}}
   ```
2. Pre-converts all audio to WAV format and measures durations
3. Bucketizes into short/long based on duration threshold
4. Processes shorts in batches, longs sequentially
5. Returns transcriptions with duration and optional word/segment timestamps in original order

## Development Commands

### Local Testing (Dry-run Mode)

```bash
# Test without loading model (macOS/local dev)
export SKIP_MODEL_LOAD=1
python test_local.py

# Or run handler directly
export SKIP_MODEL_LOAD=1
python handler.py
```

### Docker Build & Run

```bash
# Build container
docker build -t parakeet-worker .

# Run locally with dry-run mode
docker run -e SKIP_MODEL_LOAD=1 -e RUNPOD_SERVERLESS=0 parakeet-worker

# Run with model loaded (requires NVIDIA GPU)
docker run --gpus all -e RUNPOD_SERVERLESS=0 parakeet-worker
```

### Install Dependencies

```bash
pip install -r requirements.txt
# For full model support (requires CUDA):
pip install "nemo_toolkit[asr]>=2.4.0" soundfile librosa
```

## Environment Variables

- `PARAKEET_MODEL`: ASR model name (default: "nvidia/parakeet-tdt-0.6b-v3")
- `SKIP_MODEL_LOAD`: Set to "1" to skip model loading for local development
- `RUNPOD_SERVERLESS`: Set to "1" to start Runpod serverless worker
- `SHORT_MAX_SEC`: Duration threshold for batching (default: 600 seconds / 10 minutes)
- `BATCH_MAX_ITEMS`: Maximum items per batch (default: 16)
- `BATCH_MAX_TOTAL_SEC`: Maximum total duration per batch (default: 1200 seconds / 20 minutes)
- `LOCAL_ATTENTION_AFTER_SEC`: Switch to local attention after this duration (default: 1440 seconds / 24 minutes)
- `HF_HOME` / `TRANSFORMERS_CACHE`: Hugging Face cache directories

## Key Considerations

- The service dynamically switches between global and local attention based on audio duration
- Short audio files are batched for efficient GPU utilization while long files are processed sequentially
- All audio is converted to 16kHz mono WAV format for consistent processing
- Temporary WAV files are cleaned up after processing
- ffmpeg and ffprobe are required system dependencies
- Results maintain original input order regardless of batching
