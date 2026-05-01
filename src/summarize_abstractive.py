"""
Abstractive summarization module (BART / T5).

Provides a unified interface for encoder-decoder transformer models
that generate new summary text rather than extracting existing sentences.
Handles long inputs via chunk-level summarization with an optional
hierarchical second pass.

Typical usage::

    from src.summarize_abstractive import AbstractiveSummarizer

    summarizer = AbstractiveSummarizer("facebook/bart-large-cnn")
    summary = summarizer.summarize(text)

    # For long transcripts:
    summary = summarizer.summarize_long(chunks, hierarchical=True)
"""

from typing import Optional

from src.utils import setup_logger

logger = setup_logger(__name__)


class AbstractiveSummarizer:
    """
    Unified abstractive summarizer for BART and T5 model families.

    Loads the model and tokenizer once at construction time. All
    subsequent calls reuse them, avoiding repeated startup cost.

    The class auto-detects whether the model is T5-based (and thus
    requires a ``"summarize: "`` prefix) by inspecting the model name.

    Args:
        model_name: HuggingFace model identifier.
                    Examples: ``"facebook/bart-large-cnn"``, ``"t5-base"``,
                    ``"google/flan-t5-base"``.
        device:     ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_t5 = "t5" in model_name.lower()

        logger.info("Loading model '%s' on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    @property
    def max_input_tokens(self) -> int:
        """Maximum input length the model was designed for."""
        name = self.model_name.lower()
        if "bart" in name:
            return 1024
        if "pegasus" in name:
            # All published Pegasus checkpoints (large / cnn_dailymail / xsum / etc.)
            # use a 1024-token max source length.
            return 1024
        # T5 and Flan-T5 default context window
        return 512

    def summarize(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 56,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """
        Summarize a single chunk of text.

        The input is truncated to :attr:`max_input_tokens` if it exceeds
        the limit. For long documents, use :meth:`summarize_long` instead.

        Args:
            text:                 Input text.
            max_length:           Maximum tokens in the generated summary.
            min_length:           Minimum tokens in the generated summary.
            num_beams:            Beam search width (higher = better but slower).
            length_penalty:       Values > 1 favor longer summaries.
            no_repeat_ngram_size: Prevents repeating N-grams of this size.

        Returns:
            Generated summary string.
        """
        import torch

        if not text or not text.strip():
            logger.warning("Empty input — returning empty summary")
            return ""

        # T5 requires a task prefix
        model_input = f"summarize: {text}" if self._is_t5 else text

        inputs = self.tokenizer(
            model_input,
            max_length=self.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        logger.info(
            "%s summarized %d → %d tokens",
            self.model_name.split("/")[-1],
            len(inputs["input_ids"][0]),
            len(summary_ids[0]),
        )

        return summary.strip()

    def summarize_long(
        self,
        chunks: list[str],
        hierarchical: bool = True,
        max_length: int = 256,
        **kwargs,
    ) -> str:
        """
        Summarize a long document provided as pre-chunked segments.

        Strategy:
            1. Summarize each chunk independently.
            2. Concatenate the per-chunk summaries.
            3. If *hierarchical* is True and the concatenation exceeds
               the model's token limit, run a second summarization pass.

        Args:
            chunks:       List of text chunks (from ``src.chunking``).
            hierarchical: Whether to apply a second-pass summary on the
                          concatenated chunk summaries.
            max_length:   Max tokens per chunk summary.
            **kwargs:     Forwarded to :meth:`summarize`.

        Returns:
            Final summary string.
        """
        if not chunks:
            return ""

        # Single chunk — no need for multi-pass
        if len(chunks) == 1:
            return self.summarize(chunks[0], max_length=max_length, **kwargs)

        # First pass: per-chunk summaries
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info("Summarizing chunk %d/%d", i + 1, len(chunks))
            summary = self.summarize(chunk, max_length=max_length, **kwargs)
            if summary:
                chunk_summaries.append(summary)

        if not chunk_summaries:
            return ""

        combined = " ".join(chunk_summaries)

        # Second pass (hierarchical) if needed
        if hierarchical:
            combined_token_count = len(
                self.tokenizer.encode(combined, add_special_tokens=False)
            )
            logger.info(
                "Combined chunk summaries: %d tokens (limit: %d)",
                combined_token_count,
                self.max_input_tokens,
            )
            return self.summarize(combined, max_length=max_length, **kwargs)

        return combined


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_summarizer(
    model_name: str = "facebook/bart-large-cnn",
    device: Optional[str] = None,
) -> AbstractiveSummarizer:
    """
    Factory function for creating a summarizer from a model name.

    This is useful for config-driven code where the model name comes
    from ``config.yaml``.

    Args:
        model_name: HuggingFace model identifier.
        device:     Compute device.

    Returns:
        Initialized :class:`AbstractiveSummarizer`.
    """
    return AbstractiveSummarizer(model_name=model_name, device=device)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Abstractive summarization")
    parser.add_argument("input", help="Path to transcript text file")
    parser.add_argument(
        "-m", "--model",
        default="facebook/bart-large-cnn",
        help="HuggingFace model name",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-length", type=int, default=56)
    parser.add_argument("-o", "--output", help="Output file path")
    args = parser.parse_args()

    from src.utils import read_text, write_text

    text = read_text(args.input)
    summarizer = AbstractiveSummarizer(model_name=args.model)
    summary = summarizer.summarize(text, max_length=args.max_length, min_length=args.min_length)

    if args.output:
        write_text(summary, args.output)
        print(f"Saved summary to {args.output}")
    else:
        print("\n=== Abstractive Summary ===\n")
        print(summary)
