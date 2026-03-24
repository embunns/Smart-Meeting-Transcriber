import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from src.utils.audio import format_timestamp, validate_audio, load_audio


class TestFormatTimestamp:
    def test_under_one_minute(self):
        assert format_timestamp(45.0) == "00:45"

    def test_one_minute(self):
        assert format_timestamp(60.0) == "01:00"

    def test_over_one_hour(self):
        assert format_timestamp(3661.0) == "01:01:01"

    def test_zero(self):
        assert format_timestamp(0.0) == "00:00"


class TestValidateAudio:
    def test_nonexistent_file(self):
        result = validate_audio("/nonexistent/file.wav")
        assert result["valid"] is False
        assert "does not exist" in result["error"]

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.write_text("dummy")
        result = validate_audio(p)
        assert result["valid"] is False
        assert "Unsupported" in result["error"]

    def test_valid_wav_file(self, tmp_path):
        audio = np.zeros(16000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, 16000)

        result = validate_audio(wav_path)
        assert result["valid"] is True
        assert result["sample_rate"] == 16000
        assert abs(result["duration"] - 1.0) < 0.1


class TestLoadAudio:
    def test_load_mono_wav(self, tmp_path):
        audio = np.random.randn(16000).astype(np.float32)
        wav_path = tmp_path / "mono.wav"
        sf.write(str(wav_path), audio, 16000)

        loaded, sr = load_audio(wav_path)
        assert sr == 16000
        assert loaded.ndim == 1
        assert len(loaded) == 16000

    def test_load_stereo_wav_returns_mono(self, tmp_path):
        audio = np.random.randn(16000, 2).astype(np.float32)
        wav_path = tmp_path / "stereo.wav"
        sf.write(str(wav_path), audio, 16000)

        loaded, sr = load_audio(wav_path)
        assert loaded.ndim == 1

    def test_file_not_found(self):
        with pytest.raises(ValueError):
            load_audio("/does/not/exist.xyz")
