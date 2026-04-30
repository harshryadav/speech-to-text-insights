"""
Transcript chunking module.

Splits long transcripts into smaller chunks that fit roughly within a
transformer model's input limit, using **word counts** as a simple proxy
for tokens (no Hugging Face download required for this step).

English text is typically ~1.2–1.4 subword tokens per word; we use a
conservative ratio so chunks stay safely under common limits (e.g. BART 1024).
"""

from typing import Optional

from src.utils import setup_logger

logger = setup_logger(__name__)


def _word_count(text: str) -> int:
    """Number of whitespace-separated words (empty / whitespace-only → 0)."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def _word_budget(max_tokens: int) -> int:
    """
    Convert a target token budget into a per-chunk word limit.

    Subword tokenizers usually produce more tokens than words; using ~0.65
    keeps chunks conservative for models like BART/T5.
    """
    return max(1, int(max_tokens * 0.65))


# ---------------------------------------------------------------------------
# Sentence-aware chunking (recommended)
# ---------------------------------------------------------------------------

def chunk_by_sentences(
    sentences: list[str],
    max_tokens: int = 800,
    overlap_sentences: int = 2,
) -> list[str]:
    """
    Group sentences into chunks that stay under an approximate token budget.

    Sentences are never split. If one sentence alone exceeds the budget, it
    still becomes its own chunk (same as before); consider shortening very
    long sentences upstream if that happens often.

    Args:
        sentences:         Sentence strings (e.g. from preprocessing).
        max_tokens:        Target max **tokens** per chunk (converted to words internally).
        overlap_sentences: Trailing sentences from the previous chunk to repeat
                           at the start of the next chunk.

    Returns:
        List of chunk strings.

    Raises:
        ValueError: If *sentences* is empty.
    """
    if not sentences:
        raise ValueError("Cannot chunk an empty sentence list")

    max_words = _word_budget(max_tokens)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        sent_words = _word_count(sentence)

        would_exceed = current_words + sent_words > max_words and current_sentences

        if would_exceed:
            chunks.append(" ".join(current_sentences))

            if overlap_sentences > 0 and len(current_sentences) > overlap_sentences:
                overlap = current_sentences[-overlap_sentences:]
                current_sentences = list(overlap)
                current_words = _word_count(" ".join(current_sentences))
            else:
                current_sentences = []
                current_words = 0

        current_sentences.append(sentence)
        current_words = _word_count(" ".join(current_sentences))

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    logger.info(
        "Chunked %d sentences into %d chunks (~%d words/chunk cap, overlap=%d)",
        len(sentences),
        len(chunks),
        max_words,
        overlap_sentences,
    )

    return chunks


# ---------------------------------------------------------------------------
# Word-window chunking (when sentences are not available)
# ---------------------------------------------------------------------------

def chunk_by_tokens(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Split plain text into overlapping chunks using a word window.

    *overlap_tokens* is treated as an approximate token overlap and converted
    to words the same way as *max_tokens*.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    max_words = _word_budget(max_tokens)
    overlap_words = max(0, _word_budget(overlap_tokens))

    if len(words) <= max_words:
        return [text.strip()]

    step = max(1, max_words - overlap_words)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step

    logger.info(
        "Word-window chunked %d words into %d chunks (cap=%d words)",
        len(words),
        len(chunks),
        max_words,
    )

    return chunks


# ---------------------------------------------------------------------------
# Convenience: pick strategy
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    sentences: Optional[list[str]] = None,
    max_tokens: int = 800,
    overlap_sentences: int = 2,
) -> list[str]:
    """
    Chunk *text*. Uses *sentences* when provided; otherwise word windows.

    If the whole *text* already fits the budget, returns ``[text]``.
    """
    if not text or not text.strip():
        return []

    max_words = _word_budget(max_tokens)
    if _word_count(text) <= max_words:
        return [text.strip()]

    if sentences:
        return chunk_by_sentences(
            sentences,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
        )
    return chunk_by_tokens(text, max_tokens=max_tokens, overlap_tokens=100)
