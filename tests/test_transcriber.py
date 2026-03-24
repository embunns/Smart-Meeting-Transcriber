import numpy as np
import pytest

from src.asr.transcriber import MeetingTranscriber, TranscriptionResult, TranscriptSegment


class TestTranscriptSegment:
    def test_basic_fields(self):
        seg = TranscriptSegment(start=0.0, end=5.0, text="Hello world")
        assert seg.start == 0.0
        assert seg.end == 5.0
        assert seg.text == "Hello world"
        assert seg.speaker is None

    def test_with_speaker(self):
        seg = TranscriptSegment(start=0.0, end=5.0, text="Hi", speaker="Speaker 1")
        assert seg.speaker == "Speaker 1"


class TestTranscriptionResult:
    def test_basic_fields(self):
        result = TranscriptionResult(full_text="Test transcript")
        assert result.full_text == "Test transcript"
        assert result.segments == []
        assert result.language is None

    def test_with_segments(self):
        segs = [
            TranscriptSegment(start=0.0, end=2.0, text="First"),
            TranscriptSegment(start=2.0, end=4.0, text="Second"),
        ]
        result = TranscriptionResult(full_text="First Second", segments=segs)
        assert len(result.segments) == 2


class TestMeetingTranscriberInit:
    def test_default_init(self):
        t = MeetingTranscriber()
        assert t.model_size == "base"
        assert t.model is None

    def test_custom_model_size(self):
        t = MeetingTranscriber(model_size="small")
        assert t.model_size == "small"

    def test_invalid_model_size(self):
        with pytest.raises(ValueError):
            MeetingTranscriber(model_size="nonexistent")

    def test_explicit_device(self):
        t = MeetingTranscriber(model_size="tiny", device="cpu")
        assert t.device == "cpu"
