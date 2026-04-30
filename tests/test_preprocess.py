"""
Tests for the preprocessing module.

Run with: pytest tests/test_preprocess.py -v
"""

import pytest

from src.preprocess import (
    remove_fillers,
    normalize_text,
    segment_sentences,
    preprocess_transcript,
)


# ---------------------------------------------------------------------------
# remove_fillers
# ---------------------------------------------------------------------------

class TestRemoveFillers:

    def test_removes_common_fillers(self):
        text = "Um so basically we need to, uh, discuss this"
        cleaned, count = remove_fillers(text)
        assert "um" not in cleaned.lower().split()
        assert "uh" not in cleaned.lower()
        assert count >= 3  # um, basically, uh

    def test_handles_empty_string(self):
        cleaned, count = remove_fillers("")
        assert cleaned == ""
        assert count == 0

    def test_handles_none_like_empty(self):
        cleaned, count = remove_fillers("   ")
        assert count == 0

    def test_preserves_meaningful_content(self):
        text = "The budget is right for this quarter."
        cleaned, count = remove_fillers(text)
        assert "budget" in cleaned
        assert "quarter" in cleaned

    def test_collapses_repeated_words(self):
        text = "We we need to to fix the the issue"
        cleaned, count = remove_fillers(text)
        assert "we we" not in cleaned.lower()

    def test_custom_filler_list(self):
        text = "Well the plan is solid honestly"
        cleaned, _ = remove_fillers(text, filler_words=["well", "honestly"])
        assert "well" not in cleaned.lower().split()
        assert "honestly" not in cleaned.lower()

    def test_returns_tuple(self):
        result = remove_fillers("um hello")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:

    def test_collapses_whitespace(self):
        text = "Hello   world  .  This is   a test ."
        result = normalize_text(text)
        assert "  " not in result

    def test_fixes_space_before_punctuation(self):
        text = "Hello . World , how are you ?"
        result = normalize_text(text)
        assert "Hello." in result
        assert "World," in result

    def test_normalizes_smart_quotes(self):
        text = "\u201cHello\u201d she said \u2018goodbye\u2019"
        result = normalize_text(text)
        assert "\u201c" not in result
        assert '"' in result
        assert "'" in result

    def test_handles_empty_input(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == "   "

    def test_normalizes_em_dash(self):
        text = "This\u2014that"
        result = normalize_text(text)
        assert "--" in result


# ---------------------------------------------------------------------------
# segment_sentences
# ---------------------------------------------------------------------------

class TestSegmentSentences:

    def test_basic_segmentation(self):
        text = "First sentence. Second sentence. Third one."
        sentences = segment_sentences(text, method="nltk")
        assert len(sentences) == 3

    def test_empty_input(self):
        assert segment_sentences("", method="nltk") == []
        assert segment_sentences("  ", method="nltk") == []

    def test_single_sentence(self):
        sentences = segment_sentences("Just one sentence.", method="nltk")
        assert len(sentences) == 1

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            segment_sentences("Hello.", method="invalid")


# ---------------------------------------------------------------------------
# preprocess_transcript (full pipeline)
# ---------------------------------------------------------------------------

class TestPreprocessTranscript:

    def test_full_pipeline(self):
        text = (
            "Um so basically what we're going to, uh, talk about "
            "today is the quarterly results. You know the numbers "
            "are looking good. I mean revenue is up."
        )
        result = preprocess_transcript(text)

        assert "original" in result
        assert "cleaned" in result
        assert "sentences" in result
        assert "stats" in result
        assert result["original"] == text
        assert result["stats"]["fillers_removed"] > 0
        assert result["stats"]["sentence_count"] > 0
        assert 0 < result["stats"]["compression_ratio"] <= 1.0

    def test_empty_input(self):
        result = preprocess_transcript("")
        assert result["cleaned"] == ""
        assert result["sentences"] == []
        assert result["stats"]["original_word_count"] == 0

    def test_no_fillers(self):
        text = "The project is on track. We will deliver next week."
        result = preprocess_transcript(text)
        assert result["stats"]["fillers_removed"] == 0
        assert result["stats"]["sentence_count"] == 2

    def test_stats_are_consistent(self):
        text = "Um hello. Uh world."
        result = preprocess_transcript(text)
        stats = result["stats"]
        assert stats["cleaned_word_count"] <= stats["original_word_count"]
        assert stats["cleaned_word_count"] > 0
