from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.asr.transcriber import MeetingTranscriber, TranscriptionResult
from src.nlp.analyzer import MeetingAnalyzer, MeetingAnalysis
from src.utils.formatter import export_to_json, export_to_txt


@dataclass
class PipelineConfig:
    asr_model: str = "base"
    summarizer_model: str = "facebook/bart-large-cnn"
    language: Optional[str] = None
    device: str = "auto"


@dataclass
class PipelineResult:
    transcript: TranscriptionResult
    analysis: MeetingAnalysis


class MeetingPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        device_id = -1
        if self.config.device == "auto":
            import torch
            device_id = 0 if torch.cuda.is_available() else -1
        elif self.config.device == "cuda":
            device_id = 0
        elif self.config.device == "cpu":
            device_id = -1

        self.transcriber = MeetingTranscriber(
            model_size=self.config.asr_model,
        )
        self.analyzer = MeetingAnalyzer(
            summarizer_model=self.config.summarizer_model,
            device=device_id,
        )

    def run(self, audio_path: str | Path) -> PipelineResult:
        audio_path = Path(audio_path)

        transcript = self.transcriber.transcribe(
            audio_path,
            language=self.config.language,
        )
        analysis = self.analyzer.analyze(transcript.full_text)

        return PipelineResult(transcript=transcript, analysis=analysis)

    def run_and_export(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        export_format: str = "json",
    ) -> PipelineResult:
        result = self.run(audio_path)

        output_dir = Path(output_dir)
        stem = Path(audio_path).stem

        if export_format == "json":
            out_path = output_dir / f"{stem}_result.json"
            export_to_json(result.transcript, result.analysis, out_path)
        elif export_format == "txt":
            out_path = output_dir / f"{stem}_result.txt"
            export_to_txt(result.transcript, result.analysis, out_path)
        else:
            raise ValueError(f"Unknown export format: {export_format}")

        return result
