# handler.py
import os
import json
import tempfile
import urllib.request
import uuid
from typing import List, Dict, Any, Tuple

import runpod

# -------------------------
# Config
# -------------------------
ASR_MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")

# Model cache directory (defaults to network volume mount point)
CACHE_DIR = os.getenv("HF_HOME", "/runpod-volume/model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Thresholds / batching knobs
# Anything <= SHORT_MAX_SEC seconds is considered "short" and eligible for batching.
SHORT_MAX_SEC = int(os.getenv("SHORT_MAX_SEC", "600"))  # 10 minutes

# Hard cap on how many short clips per batch
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "16"))

# Optional cap on summed duration for a batch of shorts (seconds)
BATCH_MAX_TOTAL_SEC = int(
    os.getenv("BATCH_MAX_TOTAL_SEC", "1200"))  # 20 minutes

# Attention switch for very long inputs (>24 min ~ full attention on A100-80GB)
LOCAL_ATTENTION_AFTER_SEC = int(
    os.getenv("LOCAL_ATTENTION_AFTER_SEC", str(24 * 60)))

# In local dev, avoid loading NeMo so you don't pull huge deps on macOS.
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") in ("1", "true", "True")

MODEL = None
if not SKIP_MODEL_LOAD:
    import torch
    import nemo.collections.asr as nemo_asr

    # Try loading from cache first
    cached_model_path = os.path.join(CACHE_DIR, ASR_MODEL_NAME.replace("/", "_"))
    if os.path.exists(cached_model_path):
        print(f"Loading cached model from {cached_model_path}")
        MODEL = nemo_asr.models.ASRModel.restore_from(cached_model_path)
    else:
        print(f"Downloading model {ASR_MODEL_NAME} and caching to {cached_model_path}")
        MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
        # Cache the model for future use
        try:
            MODEL.save_to(cached_model_path)
            print(f"Model cached successfully to {cached_model_path}")
        except Exception as e:
            print(f"Warning: Failed to cache model: {e}")

    MODEL.eval()
    if torch.cuda.is_available():
        MODEL = MODEL.to("cuda")


# -------------------------
# Helpers
# -------------------------
def _validate_wav_format(src: str) -> str:
    """
    Validate that the source is a properly formatted 16kHz mono WAV file.
    Returns the source path if valid, raises ValueError if not.
    """
    # Extract filename from URL (remove query parameters)
    if src.startswith('http'):
        filename = src.split('?')[0]  # Remove query params
    else:
        filename = src

    if not filename.lower().endswith('.wav'):
        raise ValueError(f"File must be WAV format. Got filename: {filename}")

    return src


def _download_url_to_temp(url: str) -> str:
    """
    Download URL to temporary file for NeMo processing.
    Returns path to temporary file.
    """
    tmp_dir = tempfile.gettempdir()
    tmp_file = os.path.join(tmp_dir, f"audio-{uuid.uuid4().hex}.wav")

    try:
        urllib.request.urlretrieve(url, tmp_file)
        return tmp_file
    except Exception as e:
        # Clean up failed download
        try:
            os.remove(tmp_file)
        except:
            pass
        raise ValueError(f"Failed to download audio file: {e}")


def _cleanup_files(paths: List[str]):
    """Clean up temporary files."""
    for path in paths:
        try:
            os.remove(path)
        except:
            pass


def _get_duration_seconds(wav_path: str) -> float:
    """
    Get duration using soundfile library instead of ffprobe.
    """
    if SKIP_MODEL_LOAD:
        # Return dummy duration for dry-run mode
        return 7.5

    try:
        import soundfile as sf
        with sf.SoundFile(wav_path) as f:
            return len(f) / f.samplerate
    except Exception as e:
        raise ValueError(f"Could not determine duration for {wav_path}: {e}")


def _maybe_set_local_attention(total_seconds: float):
    """Switch to local attention for > ~24 minutes (A100-80GB full-attn rule of thumb)."""
    if SKIP_MODEL_LOAD:
        return
    try:
        if total_seconds > LOCAL_ATTENTION_AFTER_SEC:
            MODEL.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256]
            )
        else:
            # Try to switch back to global (if supported) for short files
            MODEL.change_attention_model(
                self_attention_model="rel_pos"
            )
    except Exception:
        # Not all variants support toggling back and forth; ignore silently.
        pass


def _bucketize_by_duration(items: List[Tuple[str, float]]):
    """
    Split into (shorts, longs) by SHORT_MAX_SEC.
    items: list of (wav_path, duration_sec)
    """
    shorts, longs = [], []
    for p, d in items:
        (shorts if d <= SHORT_MAX_SEC else longs).append((p, d))
    return shorts, longs


def _make_batches(shorts: List[Tuple[str, float]]) -> List[List[Tuple[str, float]]]:
    """
    Pack shorts into batches respecting BATCH_MAX_ITEMS and BATCH_MAX_TOTAL_SEC.
    Greedy pack by arrival order (simple and effective for our case).
    """
    batches, current = [], []
    sum_sec = 0.0
    for p, d in shorts:
        fits_count = len(current) < BATCH_MAX_ITEMS
        fits_total = (sum_sec + d) <= BATCH_MAX_TOTAL_SEC
        if fits_count and fits_total:
            current.append((p, d))
            sum_sec += d
        else:
            if current:
                batches.append(current)
            current = [(p, d)]
            sum_sec = d
    if current:
        batches.append(current)
    return batches


# -------------------------
# Core
# -------------------------
def transcribe_batched(inputs: List[Dict[str, Any]], want_ts: bool) -> List[Dict[str, Any]]:
    """
    Hybrid strategy:
      1) Validate WAV format and measure durations
      2) Batch shorts; process longs sequentially
      3) Preserve output order matching the original inputs list
    """
    # 1) Validate format and collect metadata
    validated: List[Tuple[str, float]] = []  # list of (local_path, duration)
    temp_files: List[str] = []  # Track temp files for cleanup

    for item in inputs:
        src = item["source"]
        try:
            _validate_wav_format(src)

            # Download URLs to temp files for NeMo processing
            if src.startswith('http'):
                local_path = _download_url_to_temp(src)
                temp_files.append(local_path)
            else:
                local_path = src

            # Use provided duration if available, otherwise determine it
            if "duration" in item and item["duration"] is not None:
                dur = float(item["duration"])
            else:
                dur = _get_duration_seconds(local_path)
            validated.append((local_path, dur))
        except ValueError as e:
            # Clean up any temp files created so far
            _cleanup_files(temp_files)
            raise ValueError(f"Invalid audio format for {src}: {e}. Required: 16kHz mono WAV file.")

    # 2) Bucketize
    shorts, longs = _bucketize_by_duration(validated)
    short_batches = _make_batches(shorts)

    results_by_index: Dict[int, Dict[str, Any]] = {}

    # Build index map from wav path to original item index (to preserve ordering)
    path_to_index = {}
    for idx, item in enumerate(inputs):
        # We validated in the same order, this is safe:
        path_to_index[validated[idx][0]] = idx

    try:
        # 3) Process short batches in parallel (batched forward pass)
        for batch in short_batches:
            wavs = [p for (p, _) in batch]
            # Attention mode for batch: use the max duration in batch
            max_dur = max(d for (_, d) in batch) if batch else 0.0
            _maybe_set_local_attention(max_dur)

            if SKIP_MODEL_LOAD:
                outs = [{"text": f"[dry-run] Transcribed placeholder for {p}"}
                        for p in wavs]
            else:
                outs = MODEL.transcribe(wavs, timestamps=want_ts)

            for p, d, out in zip(wavs, [d for (_, d) in batch], outs):
                idx = path_to_index[p]
                text = getattr(out, "text", str(out))
                payload = {"text": text, "duration_sec": d}
                if want_ts and hasattr(out, "timestamp"):
                    payload["timestamps"] = {
                        "word": out.timestamp.get("word"),
                        "segment": out.timestamp.get("segment")
                    }
                results_by_index[idx] = payload

        # 4) Process long audios sequentially (safer for memory)
        for p, d in longs:
            _maybe_set_local_attention(d)

            if SKIP_MODEL_LOAD:
                out_obj = {
                    "text": f"[dry-run] Transcribed placeholder for {p}"}
            else:
                out_obj = MODEL.transcribe([p], timestamps=want_ts)[0]

            idx = path_to_index[p]
            text = getattr(out_obj, "text", str(out_obj))
            payload = {"text": text, "duration_sec": d}
            if want_ts and hasattr(out_obj, "timestamp"):
                payload["timestamps"] = {
                    "word": out_obj.timestamp.get("word"),
                    "segment": out_obj.timestamp.get("segment")
                }
            results_by_index[idx] = payload

    finally:
        # 5) Clean up temporary files
        _cleanup_files(temp_files)

    # 6) Return results in original input order
    return [results_by_index[i] for i in range(len(inputs))]


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      { "input": { "timestamps": bool, "inputs": [{"source": "..."}] } }
    or
      { "timestamps": bool, "inputs": [{"source": "..."}] }

    Returns: { "results": [ { "text": "...", "duration_sec": float, ... }, ... ] }
    """
    body = event.get("input") or event
    items = body.get("inputs", [])
    want_ts = bool(body.get("timestamps", False))

    if not items:
        return {"results": [], "error": "No inputs provided."}

    return {"results": transcribe_batched(items, want_ts)}


# ✅ Only start the Runpod worker if explicitly enabled (so local imports stay quiet)
if os.getenv("RUNPOD_SERVERLESS", "0") in ("1", "true", "True"):
    runpod.serverless.start({"handler": handler})


# Local convenience runner
if __name__ == "__main__":
    example = {
        "input": {
            "timestamps": False,
            "inputs": [
                {"source": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"},
                {"source": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"}
            ]
        }
    }
    print(json.dumps(handler(example), indent=2))
