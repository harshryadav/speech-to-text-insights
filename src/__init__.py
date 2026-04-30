"""
Speech-to-Text Insights — Core Package

End-to-end pipeline for automatic speech transcription and summarization.

Modules:
    utils                  — Config loading, logging, reproducibility, I/O helpers
    transcribe             — Whisper ASR wrapper for audio-to-text
    preprocess             — Filler removal, normalization, sentence segmentation
    chunking               — Sentence-aware chunking for token-limited models
    summarize_extractive   — TextRank extractive summarization (baseline)
    summarize_abstractive  — BART / T5 abstractive summarization
    evaluate               — ROUGE, BERTScore, and human evaluation helpers
    pipeline               — End-to-end orchestrator tying all stages together
"""

__version__ = "0.1.0"
