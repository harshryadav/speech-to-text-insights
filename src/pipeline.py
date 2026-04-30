"""
End-to-end pipeline orchestrator.

Ties together transcription, preprocessing, chunking, summarization,
and evaluation into a single callable pipeline. This is the main
entry point for running the full system programmatically.

Typical usage::

    from src.pipeline import SpeechInsightsPipeline

    pipeline = SpeechInsightsPipeline.from_config("configs/config.yaml")
    result = pipeline.run("data/raw/meeting.wav")
    print(result["summaries"]["bart"])
"""

import time
from pathlib import Path
from typing import Any, Optional, Union

from src.utils import load_config, get_nested, set_seed, setup_logger, write_json

logger = setup_logger(__name__)


class SpeechInsightsPipeline:
    """
    Orchestrates the full audio → transcript → summary → evaluation flow.

    Initializes all sub-components lazily (on first use) to avoid loading
    heavy models until they're actually needed.

    Args:
        config: Parsed configuration dictionary.
    """

    def __init__(self, config: dict):
        self.config = config

        seed = config.get("seed", 42)
        set_seed(seed)

        # Lazy-loaded components (initialized on first use)
        self._transcriber = None
        self._preprocessor_config = config.get("preprocessing", {})
        self._chunking_config = config.get("chunking", {})
        self._summarizers: dict = {}  # model_name → AbstractiveSummarizer

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "configs/config.yaml") -> "SpeechInsightsPipeline":
        """
        Create a pipeline from a YAML config file.

        Args:
            config_path: Path to the config file.

        Returns:
            Initialized pipeline instance.
        """
        config = load_config(config_path)
        return cls(config)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    @property
    def transcriber(self):
        """Lazy-load the Whisper transcriber on first access."""
        if self._transcriber is None:
            from src.transcribe import WhisperTranscriber

            whisper_cfg = self.config.get("whisper", {})
            self._transcriber = WhisperTranscriber(
                model_size=whisper_cfg.get("model_size", "base"),
                language=whisper_cfg.get("language", "en"),
            )
        return self._transcriber

    def _get_summarizer(self, model_name: str):
        """Lazy-load an abstractive summarizer by model name."""
        if model_name not in self._summarizers:
            from src.summarize_abstractive import AbstractiveSummarizer

            self._summarizers[model_name] = AbstractiveSummarizer(model_name=model_name)
        return self._summarizers[model_name]

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Union[str, Path]) -> dict:
        """
        Stage 1: Transcribe audio to text using Whisper.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Transcript dict with ``text``, ``segments``, ``duration_seconds``, etc.
        """
        logger.info("=== Stage 1: Transcription ===")
        return self.transcriber.transcribe(audio_path)

    def preprocess(self, text: str) -> dict:
        """
        Stage 2: Clean and segment the raw transcript.

        Args:
            text: Raw transcript text.

        Returns:
            Preprocessing result dict with ``cleaned``, ``sentences``, ``stats``.
        """
        logger.info("=== Stage 2: Preprocessing ===")
        from src.preprocess import preprocess_transcript

        return preprocess_transcript(
            text,
            filler_words=self._preprocessor_config.get("filler_words"),
            sentence_method=self._preprocessor_config.get("sentence_method", "nltk"),
        )

    def chunk(self, text: str, sentences: Optional[list[str]] = None) -> list[str]:
        """
        Stage 3: Split text into model-friendly chunks.

        Short-circuits if the text already fits within the token budget.

        Args:
            text:      Full cleaned text.
            sentences: Pre-segmented sentences (preferred for better chunks).

        Returns:
            List of chunk strings.
        """
        logger.info("=== Stage 3: Chunking ===")
        from src.chunking import chunk_text

        return chunk_text(
            text=text,
            sentences=sentences,
            max_tokens=self._chunking_config.get("max_chunk_tokens", 800),
            overlap_sentences=self._chunking_config.get("overlap_sentences", 2),
        )

    def summarize(
        self,
        text: str,
        chunks: list[str],
        methods: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """
        Stage 4: Generate summaries using specified methods.

        Args:
            text:    Full cleaned transcript (for extractive methods).
            chunks:  Chunked text (for abstractive methods).
            methods: List of method names to run. Defaults to
                     ``["textrank", "bart", "t5"]``.

        Returns:
            Dict mapping method name to summary string.
        """
        logger.info("=== Stage 4: Summarization ===")

        methods = methods or ["textrank", "bart", "t5"]
        summaries = {}
        summ_cfg = self.config.get("summarization", {})
        hierarchical = summ_cfg.get("hierarchical", True)

        for method in methods:
            try:
                start = time.perf_counter()

                if method == "textrank":
                    summaries[method] = self._run_textrank(text, summ_cfg)

                elif method in ("bart", "t5"):
                    summaries[method] = self._run_abstractive(
                        method, chunks, summ_cfg, hierarchical
                    )

                else:
                    logger.warning("Unknown method '%s' — skipping", method)
                    continue

                elapsed = time.perf_counter() - start
                logger.info(
                    "%s summary: %d words in %.2fs",
                    method,
                    len(summaries[method].split()),
                    elapsed,
                )

            except Exception as e:
                logger.error("Failed to run %s: %s", method, e)
                summaries[method] = f"[Error: {e}]"

        return summaries

    def evaluate(
        self,
        summaries: dict[str, str],
        reference: str,
    ) -> dict[str, dict]:
        """
        Stage 5: Evaluate generated summaries against a reference.

        Args:
            summaries: Dict of method_name → summary text.
            reference: Gold-standard reference summary.

        Returns:
            Dict mapping method names to their ROUGE score dicts.
        """
        logger.info("=== Stage 5: Evaluation ===")
        from src.evaluate import compute_rouge, compression_ratio

        results = {}
        for method, summary in summaries.items():
            if summary.startswith("[Error"):
                results[method] = {"error": summary}
                continue

            try:
                rouge = compute_rouge(summary, reference)
                rouge["compression_ratio"] = compression_ratio(reference, summary)
                results[method] = rouge
            except ValueError as e:
                logger.warning("Cannot evaluate %s: %s", method, e)
                results[method] = {"error": str(e)}

        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        audio_path: Union[str, Path],
        reference_summary: Optional[str] = None,
        methods: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Run the complete pipeline: audio → transcript → clean → chunk → summarize → evaluate.

        Args:
            audio_path:        Path to the input audio file.
            reference_summary: Optional reference summary for evaluation.
            methods:           Summarization methods to run.

        Returns:
            Comprehensive result dict with keys:

            - ``transcript``: Raw Whisper output.
            - ``preprocessing``: Cleaned text, sentences, stats.
            - ``chunks``: List of chunked text segments.
            - ``summaries``: Dict of method → summary text.
            - ``evaluation``: Dict of method → ROUGE scores (if reference given).
            - ``timings``: Wall-clock time for each stage.
        """
        logger.info("=" * 60)
        logger.info("Starting pipeline for: %s", audio_path)
        logger.info("=" * 60)

        timings = {}
        pipeline_start = time.perf_counter()

        # Stage 1: Transcription
        t0 = time.perf_counter()
        transcript = self.transcribe(audio_path)
        timings["transcription"] = round(time.perf_counter() - t0, 2)

        # Stage 2: Preprocessing
        t0 = time.perf_counter()
        preprocessing = self.preprocess(transcript["text"])
        timings["preprocessing"] = round(time.perf_counter() - t0, 2)

        # Stage 3: Chunking
        t0 = time.perf_counter()
        chunks = self.chunk(preprocessing["cleaned"], preprocessing["sentences"])
        timings["chunking"] = round(time.perf_counter() - t0, 2)

        # Stage 4: Summarization
        t0 = time.perf_counter()
        summaries = self.summarize(preprocessing["cleaned"], chunks, methods)
        timings["summarization"] = round(time.perf_counter() - t0, 2)

        # Stage 5: Evaluation (if reference is provided)
        evaluation = None
        if reference_summary:
            t0 = time.perf_counter()
            evaluation = self.evaluate(summaries, reference_summary)
            timings["evaluation"] = round(time.perf_counter() - t0, 2)

        timings["total"] = round(time.perf_counter() - pipeline_start, 2)

        logger.info("Pipeline complete in %.2fs", timings["total"])

        return {
            "audio_file": str(audio_path),
            "transcript": transcript,
            "preprocessing": preprocessing,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "summaries": summaries,
            "evaluation": evaluation,
            "timings": timings,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_textrank(self, text: str, summ_cfg: dict) -> str:
        from src.summarize_extractive import textrank_summarize

        ext_cfg = summ_cfg.get("extractive", {})
        return textrank_summarize(
            text,
            num_sentences=ext_cfg.get("num_sentences", 5),
            language=ext_cfg.get("language", "english"),
        )

    def _run_abstractive(
        self, method: str, chunks: list[str], summ_cfg: dict, hierarchical: bool
    ) -> str:
        abs_cfg = get_nested(summ_cfg, "abstractive", method) or {}
        model_name = abs_cfg.get("model_name", {
            "bart": "facebook/bart-large-cnn",
            "t5": "google/flan-t5-base",
        }.get(method, "facebook/bart-large-cnn"))

        summarizer = self._get_summarizer(model_name)

        return summarizer.summarize_long(
            chunks,
            hierarchical=hierarchical,
            max_length=abs_cfg.get("max_summary_length", 256),
            min_length=abs_cfg.get("min_summary_length", 56),
            num_beams=abs_cfg.get("num_beams", 4),
            length_penalty=abs_cfg.get("length_penalty", 2.0),
            no_repeat_ngram_size=abs_cfg.get("no_repeat_ngram_size", 3),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the full Speech Insights pipeline")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("-c", "--config", default="configs/config.yaml")
    parser.add_argument("-r", "--reference", help="Path to reference summary file")
    parser.add_argument("-o", "--output", help="Output JSON path for results")
    parser.add_argument(
        "-m", "--methods", nargs="+",
        default=["textrank", "bart", "t5"],
        help="Summarization methods to run",
    )
    args = parser.parse_args()

    from src.utils import read_text

    pipeline = SpeechInsightsPipeline.from_config(args.config)

    ref = read_text(args.reference) if args.reference else None
    result = pipeline.run(args.audio, reference_summary=ref, methods=args.methods)

    # Print summaries
    for method, summary in result["summaries"].items():
        print(f"\n{'=' * 40}")
        print(f"  {method.upper()} Summary")
        print(f"{'=' * 40}")
        print(summary)

    # Print evaluation
    if result["evaluation"]:
        print(f"\n{'=' * 40}")
        print("  Evaluation Results")
        print(f"{'=' * 40}")
        print(json.dumps(result["evaluation"], indent=2))

    # Save full results
    if args.output:
        # Remove non-serializable parts for JSON output
        serializable = {
            k: v for k, v in result.items()
            if k != "transcript"  # transcript can be very large
        }
        serializable["transcript_text"] = result["transcript"]["text"][:500] + "..."
        write_json(serializable, args.output)
        print(f"\nFull results saved to {args.output}")
