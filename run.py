import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import MeetingPipeline, PipelineConfig
from src.utils.formatter import format_transcript, format_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting audio file and extract action items.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py meeting.wav
  python run.py meeting.mp3 --model small --export json --output-dir ./results
  python run.py meeting.wav --language en --no-timestamps
        """,
    )

    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language code, e.g. 'en', 'id'. Auto-detect if not set.",
    )
    parser.add_argument(
        "--export",
        choices=["json", "txt"],
        default=None,
        help="Export format. If not set, prints to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for exported files (default: ./output)",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Omit timestamps from transcript output",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        asr_model=args.model,
        language=args.language,
    )
    pipeline = MeetingPipeline(config)

    print(f"processing: {audio_path.name}")

    if args.export:
        result = pipeline.run_and_export(
            audio_path=audio_path,
            output_dir=args.output_dir,
            export_format=args.export,
        )
        stem = audio_path.stem
        out_file = Path(args.output_dir) / f"{stem}_result.{args.export}"
        print(f"saved: {out_file}")
    else:
        result = pipeline.run(audio_path)

        print("\n--- TRANSCRIPT ---")
        print(format_transcript(result.transcript, with_timestamps=not args.no_timestamps))

        print("\n--- ANALYSIS ---")
        print(format_analysis(result.analysis))


if __name__ == "__main__":
    main()
