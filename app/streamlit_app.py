import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.transcriber import MeetingTranscriber
from src.nlp.analyzer import MeetingAnalyzer
from src.utils.audio import validate_audio
from src.utils.formatter import format_transcript, format_analysis, export_to_json


st.set_page_config(
    page_title="Meeting Transcriber",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: -0.02em;
    }

    .metric-container {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 1rem 1.25rem;
    }

    .tag {
        display: inline-block;
        background: #111;
        color: #fff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        padding: 0.2rem 0.6rem;
        border-radius: 2px;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .action-item {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        border-left: 3px solid #111;
        padding-left: 0.75rem;
        margin-bottom: 0.5rem;
        color: #222;
    }

    .stTextArea textarea {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_transcriber(model_size: str) -> MeetingTranscriber:
    return MeetingTranscriber(model_size=model_size).load_model()


@st.cache_resource(show_spinner=False)
def load_analyzer() -> MeetingAnalyzer:
    return MeetingAnalyzer()


with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    model_size = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower.",
    )

    show_timestamps = st.toggle("Show timestamps", value=True)
    export_results = st.toggle("Export results as JSON", value=False)

    st.markdown("---")
    st.markdown(
        """
        **Models used**
        - ASR: [OpenAI Whisper](https://github.com/openai/whisper)
        - Summarization: facebook/bart-large-cnn
        """,
        unsafe_allow_html=False,
    )


st.markdown("# Meeting Transcriber")
st.markdown(
    "Upload a meeting recording to get a full transcript, automatic summary, and extracted action items."
)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    label_visibility="visible",
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    validation = validate_audio(tmp_path)

    if not validation["valid"]:
        st.error(f"Invalid audio file: {validation['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{validation['duration']:.1f}s")
        with col2:
            st.metric("Sample rate", f"{validation['sample_rate']} Hz")
        with col3:
            st.metric("Channels", validation["channels"])

        st.markdown("---")

        run = st.button("Run transcription", type="primary", use_container_width=False)

        if run:
            with st.spinner("Loading ASR model..."):
                transcriber = load_transcriber(model_size)

            with st.spinner("Transcribing audio..."):
                transcript = transcriber.transcribe(tmp_path)

            with st.spinner("Analyzing transcript..."):
                analyzer = load_analyzer()
                analysis = analyzer.analyze(transcript.full_text)

            st.session_state["transcript"] = transcript
            st.session_state["analysis"] = analysis

    if "transcript" in st.session_state and "analysis" in st.session_state:
        transcript = st.session_state["transcript"]
        analysis = st.session_state["analysis"]

        tab1, tab2, tab3 = st.tabs(["Transcript", "Summary", "Action Items"])

        with tab1:
            st.markdown("### Transcript")
            formatted = format_transcript(transcript, with_timestamps=show_timestamps)
            st.text_area(
                label="",
                value=formatted,
                height=400,
                label_visibility="collapsed",
            )
            if transcript.language:
                st.caption(f"Detected language: {transcript.language}")

        with tab2:
            st.markdown("### Summary")
            if analysis.summary:
                st.markdown(analysis.summary)
            else:
                st.info("No summary generated. The transcript may be too short.")

            st.markdown("**Key topics**")
            if analysis.key_topics:
                tags_html = "".join(
                    f'<span class="tag">{t}</span>' for t in analysis.key_topics
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.caption("No topics detected.")

        with tab3:
            st.markdown("### Action Items")
            if analysis.action_items:
                for item in analysis.action_items:
                    st.markdown(
                        f'<div class="action-item">{item}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No action items detected in this transcript.")

        if export_results:
            import json, io

            payload = {
                "transcript": {
                    "full_text": transcript.full_text,
                    "language": transcript.language,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in transcript.segments
                    ],
                },
                "analysis": {
                    "summary": analysis.summary,
                    "action_items": analysis.action_items,
                    "key_topics": analysis.key_topics,
                    "word_count": analysis.word_count,
                },
            }
            json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                label="Download JSON",
                data=json_bytes,
                file_name="meeting_result.json",
                mime="application/json",
            )

    tmp_path.unlink(missing_ok=True)
