import re
from dataclasses import dataclass, field
from typing import Optional

from transformers import pipeline


ACTION_PATTERNS = [
    r"\b(will|should|must|needs? to|going to|plan(?:s|ned)? to|assigned to|responsible for)\b",
    r"\b(follow[- ]up|action item|task|todo|deadline|by (?:monday|tuesday|wednesday|thursday|friday|eod|eow))\b",
    r"\b(send|prepare|review|submit|update|create|schedule|contact|reach out|coordinate|finalize)\b.{0,60}\b(by|before|until|this week|next week|tomorrow)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ACTION_PATTERNS]


@dataclass
class MeetingAnalysis:
    summary: str
    action_items: list[str] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)
    word_count: int = 0


class MeetingAnalyzer:
    DEFAULT_SUMMARIZER = "facebook/bart-large-cnn"

    def __init__(self, summarizer_model: Optional[str] = None, device: int = -1):
        self.summarizer_model = summarizer_model or self.DEFAULT_SUMMARIZER
        self.device = device
        self._summarizer = None

    def _load_summarizer(self):
        if self._summarizer is None:
            self._summarizer = pipeline(
                "summarization",
                model=self.summarizer_model,
                device=self.device,
            )
        return self._summarizer

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        if not text or not text.strip():
            return ""

        word_count = len(text.split())
        if word_count < 30:
            return text.strip()

        summarizer = self._load_summarizer()

        chunks = self._chunk_text(text, max_tokens=900)

        summaries = []
        for chunk in chunks:
            result = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            summaries.append(result[0]["summary_text"])

        if len(summaries) == 1:
            return summaries[0]

        combined = " ".join(summaries)
        final = summarizer(
            combined,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        return final[0]["summary_text"]

    def extract_action_items(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        action_items = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for pattern in COMPILED_PATTERNS:
                if pattern.search(sentence):
                    cleaned = re.sub(r"\s+", " ", sentence)
                    if cleaned not in action_items:
                        action_items.append(cleaned)
                    break

        return action_items

    def extract_key_topics(self, text: str, top_n: int = 5) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

        stopwords = {
            "that", "this", "with", "have", "will", "from", "they",
            "been", "were", "said", "also", "just", "some", "what",
            "when", "then", "than", "into", "more", "about", "would",
            "could", "should", "their", "there", "here", "which",
        }

        freq: dict[str, int] = {}
        for word in words:
            if word not in stopwords:
                freq[word] = freq.get(word, 0) + 1

        sorted_words = sorted(freq, key=freq.get, reverse=True)
        return sorted_words[:top_n]

    def analyze(self, transcript: str) -> MeetingAnalysis:
        summary = self.summarize(transcript)
        action_items = self.extract_action_items(transcript)
        key_topics = self.extract_key_topics(transcript)
        word_count = len(transcript.split())

        return MeetingAnalysis(
            summary=summary,
            action_items=action_items,
            key_topics=key_topics,
            word_count=word_count,
        )

    @staticmethod
    def _chunk_text(text: str, max_tokens: int = 900) -> list[str]:
        words = text.split()
        chunks = []
        current_chunk: list[str] = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
