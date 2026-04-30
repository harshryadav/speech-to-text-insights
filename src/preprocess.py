"""
Transcript preprocessing module.

Cleans raw Whisper transcripts by removing filler words, normalizing
whitespace/punctuation, and segmenting text into sentences. Each step
is independently callable and the full pipeline is available via
:func:`preprocess_transcript`.

Typical usage::

    from src.preprocess import preprocess_transcript

    result = preprocess_transcript(raw_text)
    print(result["cleaned"])
    print(result["stats"]["fillers_removed"])
"""

import re
from typing import Optional

from src.utils import setup_logger

logger = setup_logger(__name__)

# Filler words and verbal tics common in spoken English.
# Sorted longest-first at match time to avoid partial-match issues.
DEFAULT_FILLERS = [
    "um", "uh", "erm", "ah", "eh",
    "like",
    "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally",
    "right", "okay so", "so yeah",
]


# ---------------------------------------------------------------------------
# Step 1: Filler Removal
# ---------------------------------------------------------------------------

def remove_fillers(
    text: str,
    filler_words: Optional[list[str]] = None,
) -> tuple[str, int]:
    """
    Remove filler words and verbal tics from transcript text.

    Uses word-boundary regex matching so legitimate uses of words like
    "right" in "the right answer" are largely preserved (though some
    false positives are expected — an acceptable tradeoff for cleaner
    transcripts).

    Also collapses repeated adjacent words (e.g. "the the" → "the")
    which are common in disfluent speech.

    Args:
        text:         Raw transcript text.
        filler_words: Custom filler list. Defaults to :data:`DEFAULT_FILLERS`.

    Returns:
        Tuple of (cleaned_text, count_of_removals).
    """
    if not text or not text.strip():
        return text, 0

    fillers = filler_words if filler_words is not None else DEFAULT_FILLERS
    fillers_sorted = sorted(fillers, key=len, reverse=True)

    count = 0
    result = text

    for filler in fillers_sorted:
        # Match the filler word optionally followed by a comma and whitespace
        pattern = r"\b" + re.escape(filler) + r"\b[,]?\s*"
        matches = re.findall(pattern, result, flags=re.IGNORECASE)
        count += len(matches)
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)

    # Collapse repeated adjacent words ("the the" → "the")
    repeated = re.findall(r"\b(\w+)\s+\1\b", result, flags=re.IGNORECASE)
    count += len(repeated)
    result = re.sub(r"\b(\w+)\s+\1\b", r"\1", result, flags=re.IGNORECASE)

    result = re.sub(r"\s+", " ", result).strip()
    return result, count


# ---------------------------------------------------------------------------
# Step 2: Text Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize whitespace, punctuation, and encoding artifacts.

    Handles:
    - Multiple consecutive spaces → single space
    - Errant spaces before punctuation (``"word ."`` → ``"word."``)
    - Unicode smart quotes / em-dashes → ASCII equivalents
    - Leading/trailing whitespace per line

    Args:
        text: Input text.

    Returns:
        Normalized text string.
    """
    if not text or not text.strip():
        return text

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Fix errant space before sentence-ending punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Normalize smart quotes and dashes to ASCII
    replacements = {
        "\u2018": "'", "\u2019": "'",   # single curly quotes
        "\u201c": '"', "\u201d": '"',   # double curly quotes
        "\u2014": " -- ", "\u2013": " - ",  # em-dash, en-dash
        "\u2026": "...",  # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Strip each line and rejoin
    lines = [line.strip() for line in text.split("\n")]
    text = " ".join(line for line in lines if line)

    return text.strip()


# ---------------------------------------------------------------------------
# Step 3: Sentence Segmentation
# ---------------------------------------------------------------------------

def segment_sentences(text: str, method: str = "nltk") -> list[str]:
    """
    Split text into individual sentences.

    Args:
        text:   Input text to segment.
        method: Segmentation backend — ``"nltk"`` (fast, recommended)
                or ``"spacy"`` (slightly better on edge cases, slower).

    Returns:
        List of sentence strings.

    Raises:
        ValueError: If *method* is not ``"nltk"`` or ``"spacy"``.
    """
    if not text or not text.strip():
        return []

    if method == "nltk":
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)

    elif method == "spacy":
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install it with: python -m spacy download en_core_web_sm"
            )
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    else:
        raise ValueError(
            f"Unknown segmentation method '{method}'. Use 'nltk' or 'spacy'."
        )


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def preprocess_transcript(
    text: str,
    filler_words: Optional[list[str]] = None,
    sentence_method: str = "nltk",
) -> dict:
    """
    Run the complete preprocessing pipeline on a raw transcript.

    Pipeline order:
        1. Filler word removal
        2. Text normalization
        3. Sentence segmentation

    Args:
        text:            Raw transcript text.
        filler_words:    Custom filler word list (or ``None`` for defaults).
        sentence_method: ``"nltk"`` or ``"spacy"`` for segmentation.

    Returns:
        Dictionary with keys:

        - ``original`` (str): The unmodified input text.
        - ``cleaned`` (str): Cleaned text after steps 1 and 2.
        - ``sentences`` (list[str]): Sentence-segmented cleaned text.
        - ``stats`` (dict): Processing statistics:

          - ``original_word_count`` (int)
          - ``cleaned_word_count`` (int)
          - ``fillers_removed`` (int)
          - ``sentence_count`` (int)
          - ``compression_ratio`` (float): cleaned / original word count.
    """
    if not text or not text.strip():
        return {
            "original": text or "",
            "cleaned": "",
            "sentences": [],
            "stats": {
                "original_word_count": 0,
                "cleaned_word_count": 0,
                "fillers_removed": 0,
                "sentence_count": 0,
                "compression_ratio": 0.0,
            },
        }

    original_word_count = len(text.split())

    # Step 1: Filler removal
    cleaned, fillers_removed = remove_fillers(text, filler_words)

    # Step 2: Normalization
    cleaned = normalize_text(cleaned)

    # Step 3: Sentence segmentation
    sentences = segment_sentences(cleaned, method=sentence_method)

    cleaned_word_count = len(cleaned.split())

    stats = {
        "original_word_count": original_word_count,
        "cleaned_word_count": cleaned_word_count,
        "fillers_removed": fillers_removed,
        "sentence_count": len(sentences),
        "compression_ratio": round(
            cleaned_word_count / max(original_word_count, 1), 4
        ),
    }

    logger.info(
        "Preprocessed: %d → %d words (-%d fillers, %d sentences, ratio=%.2f)",
        original_word_count,
        cleaned_word_count,
        fillers_removed,
        len(sentences),
        stats["compression_ratio"],
    )

    return {
        "original": text,
        "cleaned": cleaned,
        "sentences": sentences,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Preprocess a transcript")
    parser.add_argument("input", help="Path to raw transcript (.txt or .json)")
    parser.add_argument("-o", "--output", help="Output path for cleaned transcript")
    parser.add_argument(
        "--method", default="nltk", choices=["nltk", "spacy"],
        help="Sentence segmentation method",
    )
    args = parser.parse_args()

    from src.utils import read_json, read_text, write_json

    if args.input.endswith(".json"):
        data = read_json(args.input)
        text = data.get("text", "")
    else:
        text = read_text(args.input)

    result = preprocess_transcript(text, sentence_method=args.method)

    if args.output:
        write_json(result, args.output)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(result["stats"], indent=2))
        print(f"\nCleaned text preview:\n{result['cleaned'][:500]}...")
