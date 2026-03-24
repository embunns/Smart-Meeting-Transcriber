import pytest

from src.nlp.analyzer import MeetingAnalyzer, MeetingAnalysis


SAMPLE_TRANSCRIPT = (
    "We discussed the Q2 revenue targets and agreed on the next steps. "
    "John will prepare the financial report by Friday. "
    "Sarah should contact the client and schedule a follow-up call. "
    "The team needs to review the product roadmap before the next sprint. "
    "We also talked about the marketing strategy for the upcoming product launch."
)


class TestMeetingAnalyzer:
    def setup_method(self):
        self.analyzer = MeetingAnalyzer()

    def test_extract_action_items_finds_items(self):
        items = self.analyzer.extract_action_items(SAMPLE_TRANSCRIPT)
        assert isinstance(items, list)
        assert len(items) > 0

    def test_extract_action_items_no_duplicates(self):
        items = self.analyzer.extract_action_items(SAMPLE_TRANSCRIPT)
        assert len(items) == len(set(items))

    def test_extract_action_items_empty_text(self):
        items = self.analyzer.extract_action_items("")
        assert items == []

    def test_extract_key_topics_returns_words(self):
        topics = self.analyzer.extract_key_topics(SAMPLE_TRANSCRIPT, top_n=5)
        assert isinstance(topics, list)
        assert 1 <= len(topics) <= 5

    def test_extract_key_topics_all_lowercase(self):
        topics = self.analyzer.extract_key_topics(SAMPLE_TRANSCRIPT)
        for t in topics:
            assert t == t.lower()

    def test_chunk_text_splits_correctly(self):
        long_text = " ".join(["word"] * 2500)
        chunks = MeetingAnalyzer._chunk_text(long_text, max_tokens=900)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk.split()) <= 900


class TestMeetingAnalysis:
    def test_defaults(self):
        analysis = MeetingAnalysis(summary="Test summary")
        assert analysis.summary == "Test summary"
        assert analysis.action_items == []
        assert analysis.key_topics == []
        assert analysis.word_count == 0
