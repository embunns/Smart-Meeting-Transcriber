import torch
import whisper
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None


class MeetingTranscriber:
    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_size must be one of {self.SUPPORTED_MODELS}")

        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_size, device=self.device)
        return self

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        self.load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        options = {
            "task": task,
            "verbose": False,
        }
        if language:
            options["language"] = language

        result = self.model.transcribe(str(audio_path), **options)

        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
            for seg in result.get("segments", [])
        ]

        return TranscriptionResult(
            full_text=result["text"].strip(),
            segments=segments,
            language=result.get("language"),
        )

    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        self.load_model()

        if sample_rate != 16000:
            audio_array = whisper.audio.resample(audio_array, sample_rate, 16000)

        audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / np.iinfo(np.int16).max

        options = {"verbose": False}
        if language:
            options["language"] = language

        result = self.model.transcribe(audio_array, **options)

        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
            for seg in result.get("segments", [])
        ]

        return TranscriptionResult(
            full_text=result["text"].strip(),
            segments=segments,
            language=result.get("language"),
        )
