import io
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".webm"}


def load_audio(
    file_path: str | Path,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    file_path = Path(file_path)

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    audio, sr = sf.read(str(file_path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def load_audio_from_bytes(
    audio_bytes: bytes,
    file_extension: str = ".wav",
) -> tuple[np.ndarray, int]:
    buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buffer, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def get_audio_duration(file_path: str | Path) -> float:
    file_path = Path(file_path)
    info = sf.info(str(file_path))
    return info.duration


def validate_audio(file_path: str | Path) -> dict:
    file_path = Path(file_path)

    if not file_path.exists():
        return {"valid": False, "error": "File does not exist"}

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {
            "valid": False,
            "error": f"Unsupported format: {file_path.suffix}",
        }

    try:
        info = sf.info(str(file_path))
        return {
            "valid": True,
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
