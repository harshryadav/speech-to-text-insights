"""
Extractive summarization module (TextRank baseline).

Implements TextRank (Mihalcea & Tarau, 2004) via the ``sumy`` library.
Selects the most important sentences from a transcript based on a
graph-ranking algorithm over sentence similarity.

This serves as the **baseline** against which BART and T5 are compared.

Typical usage::

    from src.summarize_extractive import textrank_summarize

    summary = textrank_summarize(cleaned_transcript, num_sentences=5)
"""

from typing import Optional

from src.utils import setup_logger

logger = setup_logger(__name__)


def textrank_summarize(
    text: str,
    num_sentences: int = 5,
    language: str = "english",
) -> str:
    """
    Produce an extractive summary using TextRank.

    Algorithm overview:
        1. Parse text into sentences.
        2. Build a similarity graph (nodes = sentences, edges = cosine
           similarity of TF-IDF vectors).
        3. Run the PageRank algorithm on the graph.
        4. Select the top-*N* ranked sentences.
        5. Return them in **original document order**.

    Args:
        text:          Input text to summarize.
        num_sentences: Number of sentences to extract.
        language:      Language for tokenization/stemming (default: ``"english"``).

    Returns:
        Summary string composed of the top-ranked sentences.
        Returns an empty string if the input is empty.
    """
    if not text or not text.strip():
        logger.warning("Empty input — returning empty summary")
        return ""

    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

    parser = PlaintextParser.from_string(text, Tokenizer(language))

    available_sentences = len(parser.document.sentences)
    if available_sentences == 0:
        logger.warning("No sentences found in input")
        return ""

    # Don't request more sentences than exist
    actual_count = min(num_sentences, available_sentences)
    if actual_count < num_sentences:
        logger.info(
            "Requested %d sentences but only %d available — using %d",
            num_sentences, available_sentences, actual_count,
        )

    stemmer = Stemmer(language)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)

    summary_sentences = summarizer(parser.document, actual_count)
    summary = " ".join(str(s) for s in summary_sentences)

    logger.info(
        "TextRank: extracted %d/%d sentences (%d → %d words)",
        actual_count,
        available_sentences,
        len(text.split()),
        len(summary.split()),
    )

    return summary


def textrank_summarize_ratio(
    text: str,
    ratio: float = 0.2,
    language: str = "english",
) -> str:
    """
    Summarize to a target proportion of the original sentence count.

    Args:
        text:     Input text.
        ratio:    Fraction of sentences to keep (0.0–1.0). Default: 20%.
        language: Language for tokenization.

    Returns:
        Extractive summary string.

    Raises:
        ValueError: If *ratio* is not in (0, 1].
    """
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer

    parser = PlaintextParser.from_string(text, Tokenizer(language))
    total = len(parser.document.sentences)
    num_sentences = max(1, int(total * ratio))

    return textrank_summarize(text, num_sentences=num_sentences, language=language)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TextRank extractive summarization")
    parser.add_argument("input", help="Path to transcript text file")
    parser.add_argument("-n", "--num-sentences", type=int, default=5)
    parser.add_argument("-r", "--ratio", type=float, default=None)
    args = parser.parse_args()

    from src.utils import read_text

    text = read_text(args.input)

    if args.ratio:
        summary = textrank_summarize_ratio(text, ratio=args.ratio)
    else:
        summary = textrank_summarize(text, num_sentences=args.num_sentences)

    print("\n=== TextRank Summary ===\n")
    print(summary)
