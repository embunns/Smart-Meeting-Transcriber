# Smart Meeting Transcriber

Converts meeting audio into structured output: full transcript, automatic summary, and extracted action items.

Built with OpenAI Whisper for speech recognition and BART for summarization.

---

## Features

- Speech-to-text transcription with timestamp segments
- Automatic meeting summary using a pre-trained summarization model
- Rule-based action item extraction from transcript text
- Web interface via Streamlit
- JSON and plain text export

## Stack

- **ASR**: OpenAI Whisper (`tiny` to `large`)
- **Summarization**: `facebook/bart-large-cnn` via Hugging Face Transformers
- **Frontend**: Streamlit
- **Audio I/O**: soundfile
- **Evaluation**: jiwer (WER)

## Project Structure

```
smart-meeting-transcriber/
├── src/
│   ├── asr/
│   │   └── transcriber.py       # Whisper wrapper, TranscriptionResult dataclass
│   ├── nlp/
│   │   └── analyzer.py          # Summarizer, action item extractor, topic extraction
│   ├── utils/
│   │   ├── audio.py             # Audio loading, validation, timestamp formatting
│   │   └── formatter.py         # Output formatting, JSON/TXT export
│   └── pipeline.py              # End-to-end pipeline class
├── app/
│   └── streamlit_app.py         # Web interface
├── tests/
│   ├── test_transcriber.py
│   ├── test_analyzer.py
│   └── test_audio_utils.py
├── notebooks/
│   └── exploration.ipynb        # Dataset loading, WER evaluation, pipeline demo
├── configs/
│   └── default.yaml
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/yourname/smart-meeting-transcriber.git
cd smart-meeting-transcriber

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

FFmpeg is required by Whisper for non-WAV formats:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Usage

**Run the web app**

```bash
streamlit run app/streamlit_app.py
```

**Use the pipeline in code**

```python
from src.pipeline import MeetingPipeline, PipelineConfig

config = PipelineConfig(asr_model="base")
pipeline = MeetingPipeline(config)

result = pipeline.run("meeting.wav")

print(result.transcript.full_text)
print(result.analysis.summary)
print(result.analysis.action_items)
```

**Export results**

```python
pipeline.run_and_export(
    audio_path="meeting.wav",
    output_dir="data/outputs",
    export_format="json",
)
```

## Running Tests

```bash
pytest tests/ -v
```

## Dataset

The exploration notebook uses [LibriSpeech ASR](https://huggingface.co/datasets/librispeech_asr) from Hugging Face for testing and WER evaluation. Any WAV/MP3/M4A/FLAC file also works as direct input.

## Model Notes

| Whisper model | Speed | Accuracy |
|---|---|---|
| tiny | fastest | lowest |
| base | fast | decent |
| small | moderate | good |
| medium | slow | very good |
| large | slowest | best |

For meeting audio with clear speech, `base` or `small` is usually sufficient.

## WER Evaluation

The notebook includes a Word Error Rate measurement using `jiwer`:

```python
from jiwer import wer
score = wer(reference_text, hypothesis_text)
```

---

## License

MIT
