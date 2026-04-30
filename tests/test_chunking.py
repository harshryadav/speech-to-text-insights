"""
Tests for the chunking module (word-budget logic, no Hugging Face).

Run with: pytest tests/test_chunking.py -v
"""

import pytest

from src.chunking import chunk_by_sentences, chunk_by_tokens, chunk_text


class TestChunkBySentences:

    def test_single_chunk_if_short(self):
        sentences = ["Hello world.", "How are you."]
        chunks = chunk_by_sentences(sentences, max_tokens=100, overlap_sentences=0)
        assert len(chunks) == 1

    def test_splits_when_exceeding_limit(self):
        sentences = [f"Sentence number {i} with some extra words." for i in range(20)]
        chunks = chunk_by_sentences(sentences, max_tokens=20, overlap_sentences=0)
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        sentences = ["Short.", "Also short.", "Tiny."]
        chunks = chunk_by_sentences(sentences, max_tokens=1000, overlap_sentences=0)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_overlap_adds_context(self):
        sentences = [f"Sentence {i}." for i in range(10)]
        chunks_no_overlap = chunk_by_sentences(
            sentences, max_tokens=10, overlap_sentences=0
        )
        chunks_with_overlap = chunk_by_sentences(
            sentences, max_tokens=10, overlap_sentences=2
        )
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_by_sentences([], max_tokens=100)

    def test_single_long_sentence_gets_own_chunk(self):
        sentences = [
            "Short.",
            "This is a very long sentence " * 20 + ".",
            "Another short one.",
        ]
        chunks = chunk_by_sentences(sentences, max_tokens=10, overlap_sentences=0)
        assert len(chunks) >= 2


class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        text = "Hello world. This is short."
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_prefers_sentences_when_available(self):
        text = "A. B. C. D. E. F. G. H. I. J."
        sentences = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J."]
        chunks = chunk_text(text, sentences=sentences, max_tokens=5)
        assert len(chunks) > 1
