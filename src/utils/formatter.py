import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.asr.transcriber import TranscriptionResult
from src.nlp.analyzer import MeetingAnalysis
from src.utils.audio import format_timestamp


def format_transcript(result: TranscriptionResult, with_timestamps: bool = True) -> str:
    if not result.segments:
        return result.full_text

    lines = []
    for seg in result.segments:
        if with_timestamps:
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            prefix = f"[{start} - {end}]"
            lines.append(f"{prefix}  {seg.text}")
        else:
            lines.append(seg.text)

    return "\n".join(lines)


def format_analysis(analysis: MeetingAnalysis) -> str:
    sections = []

    sections.append("SUMMARY\n" + "-" * 40)
    sections.append(analysis.summary or "No summary generated.")

    sections.append("\nACTION ITEMS\n" + "-" * 40)
    if analysis.action_items:
        for item in analysis.action_items:
            sections.append(f"  - {item}")
    else:
        sections.append("  No action items detected.")

    sections.append("\nKEY TOPICS\n" + "-" * 40)
    if analysis.key_topics:
        sections.append("  " + ", ".join(analysis.key_topics))
    else:
        sections.append("  No topics extracted.")

    sections.append(f"\nWord count: {analysis.word_count}")

    return "\n".join(sections)


def export_to_json(
    transcript: TranscriptionResult,
    analysis: MeetingAnalysis,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "transcript": {
            "full_text": transcript.full_text,
            "language": transcript.language,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in transcript.segments
            ],
        },
        "analysis": {
            "summary": analysis.summary,
            "action_items": analysis.action_items,
            "key_topics": analysis.key_topics,
            "word_count": analysis.word_count,
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def export_to_txt(
    transcript: TranscriptionResult,
    analysis: MeetingAnalysis,
    output_path: str | Path,
    with_timestamps: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = []
    content.append("MEETING TRANSCRIPT\n" + "=" * 60)
    content.append(format_transcript(transcript, with_timestamps=with_timestamps))
    content.append("\n" + "=" * 60)
    content.append(format_analysis(analysis))

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(content))
