"""MERT embedder + audio download + spectrogram fingerprint helpers.

Imports torch/transformers lazily so that lite-mode commands (seed-from-catalog
queries) don't pay the model-load cost.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np


MODEL_ID = "m-a-p/MERT-v1-95M"
SR_MERT = 24000
SR_FP = 22050
MAX_SECONDS = 180
N_MELS = 128
HOP_LENGTH = 512
FP_SIZE = 224
MERT_OFFSET_S = 15
MERT_WINDOW_S = 60


def _device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


_model_cache: dict = {}


def load_mert():
    """Load MERT once, cache. Heavy import: only call from full-mode paths."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["processor"]
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    device = _device()
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model_cache["model"] = model
    _model_cache["processor"] = processor
    _model_cache["device"] = device
    return model, processor


def download_audio(url: str, out_path: Path) -> Path:
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
           "--no-playlist", "--quiet", "--no-warnings",
           "-o", str(out_path.with_suffix(".%(ext)s")), url]
    subprocess.run(cmd, check=True)
    return out_path.with_suffix(".wav")


def embed_audio_file(wav: Path) -> np.ndarray:
    """Run MERT on a wav file, return a normalized 768-dim vector."""
    import librosa
    import torch

    model, processor = load_mert()
    device = _model_cache["device"]
    y, _ = librosa.load(wav, sr=SR_MERT, offset=MERT_OFFSET_S,
                        duration=MERT_WINDOW_S, mono=True)
    inputs = processor(y, sampling_rate=SR_MERT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    vec = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return (vec / (np.linalg.norm(vec) + 1e-9)).astype(np.float32)


def embed_url(url: str) -> tuple[np.ndarray, float]:
    """Download a YouTube URL, embed with MERT, return (vector, bpm).
    Audio is deleted after."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = download_audio(url, Path(tmp) / "track")
        bpm, _ = render_fingerprint(wav)
        vec = embed_audio_file(wav)
    return vec, bpm


def render_fingerprint(wav: Path) -> tuple[float, np.ndarray]:
    """Return (bpm, 224x224 uint8 mel-spectrogram fingerprint)."""
    import librosa
    from PIL import Image

    y, _ = librosa.load(wav, sr=SR_FP, duration=MAX_SECONDS, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR_FP)
    bpm = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR_FP, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    fp_full = (norm * 255).astype(np.uint8)
    img = Image.fromarray(fp_full, mode="L").resize(
        (FP_SIZE, FP_SIZE), Image.LANCZOS
    )
    return bpm, np.array(img)
