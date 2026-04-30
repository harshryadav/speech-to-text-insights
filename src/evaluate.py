"""
Evaluation module for summarization quality.

Provides automatic metrics (ROUGE, BERTScore) and helpers for human
evaluation. All functions work on single pairs or batches.

Typical usage::

    from src.evaluate import compute_rouge, evaluate_batch

    scores = compute_rouge(predicted_summary, reference_summary)
    batch_scores = evaluate_batch(predictions, references)
"""

from typing import Optional

import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    prediction: str,
    reference: str,
    rouge_types: Optional[list[str]] = None,
) -> dict:
    """
    Compute ROUGE scores between a predicted and reference summary.

    Args:
        prediction:  Model-generated summary text.
        reference:   Human-written reference summary text.
        rouge_types: Which ROUGE variants to compute.
                     Defaults to ``["rouge1", "rouge2", "rougeL"]``.

    Returns:
        Nested dict ``{metric: {precision, recall, f1}}``.
        All values are rounded to 4 decimal places.

    Raises:
        ValueError: If either input is empty.
    """
    from rouge_score import rouge_scorer

    if not prediction or not prediction.strip():
        raise ValueError("Prediction text is empty — cannot compute ROUGE")
    if not reference or not reference.strip():
        raise ValueError("Reference text is empty — cannot compute ROUGE")

    types = rouge_types or ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(types, use_stemmer=True)

    # NOTE: rouge_scorer.score(target, prediction) — reference first
    scores = scorer.score(reference, prediction)

    return {
        metric: {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "f1": round(score.fmeasure, 4),
        }
        for metric, score in scores.items()
    }


def compute_rouge_f1(prediction: str, reference: str) -> dict[str, float]:
    """
    Shorthand: return only F1 scores for the three standard ROUGE variants.

    Args:
        prediction: Generated summary.
        reference:  Reference summary.

    Returns:
        ``{"rouge1": float, "rouge2": float, "rougeL": float}``
    """
    full = compute_rouge(prediction, reference)
    return {metric: vals["f1"] for metric, vals in full.items()}


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "roberta-large",
    device: Optional[str] = None,
) -> dict[str, float]:
    """
    Compute BERTScore (semantic similarity via contextual embeddings).

    BERTScore captures meaning overlap even when surface-level wording
    differs — complementing ROUGE's n-gram overlap approach.

    Args:
        predictions: List of generated summaries.
        references:  List of reference summaries (same length).
        model_type:  Model to use for embedding (default: ``roberta-large``).
        device:      ``"cuda"`` or ``"cpu"`` (auto-detected if ``None``).

    Returns:
        Dict with average ``precision``, ``recall``, and ``f1`` across
        all pairs, rounded to 4 decimal places.

    Raises:
        ValueError: If list lengths don't match or are empty.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    if not predictions:
        raise ValueError("Empty input lists")

    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        device=device,
        verbose=False,
    )

    result = {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }

    logger.info("BERTScore — P=%.4f  R=%.4f  F1=%.4f", result["precision"], result["recall"], result["f1"])
    return result


# ---------------------------------------------------------------------------
# Batch Evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    predictions: list[str],
    references: list[str],
    rouge_types: Optional[list[str]] = None,
    include_bertscore: bool = False,
    bertscore_model: str = "roberta-large",
) -> dict:
    """
    Evaluate a batch of prediction/reference pairs.

    Computes per-sample ROUGE scores, then averages. Optionally includes
    BERTScore. Returns both per-sample and aggregate results.

    Args:
        predictions:      List of generated summaries.
        references:       List of reference summaries.
        rouge_types:      ROUGE variants (default: rouge1, rouge2, rougeL).
        include_bertscore: Whether to compute BERTScore (slower).
        bertscore_model:  Model for BERTScore embeddings.

    Returns:
        Dictionary with:

        - ``per_sample`` (list[dict]): ROUGE scores for each pair.
        - ``average`` (dict): Mean ROUGE F1 scores across the batch.
        - ``bertscore`` (dict, optional): Average BERTScore if requested.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    types = rouge_types or ["rouge1", "rouge2", "rougeL"]

    # Per-sample ROUGE
    per_sample = []
    for pred, ref in zip(predictions, references):
        try:
            scores = compute_rouge(pred, ref, rouge_types=types)
            per_sample.append(scores)
        except ValueError as e:
            logger.warning("Skipping sample: %s", e)
            per_sample.append(None)

    # Average ROUGE F1
    valid_samples = [s for s in per_sample if s is not None]
    average = {}
    for metric in types:
        f1_values = [s[metric]["f1"] for s in valid_samples]
        average[metric] = round(sum(f1_values) / max(len(f1_values), 1), 4)

    result = {
        "per_sample": per_sample,
        "average": average,
        "num_samples": len(predictions),
        "num_valid": len(valid_samples),
    }

    # Optional BERTScore
    if include_bertscore:
        valid_preds = [p for p, s in zip(predictions, per_sample) if s is not None]
        valid_refs = [r for r, s in zip(references, per_sample) if s is not None]
        if valid_preds:
            result["bertscore"] = compute_bertscore(
                valid_preds, valid_refs, model_type=bertscore_model
            )

    logger.info(
        "Batch evaluation: %d/%d valid | avg ROUGE-1=%.4f  ROUGE-2=%.4f  ROUGE-L=%.4f",
        len(valid_samples),
        len(predictions),
        average.get("rouge1", 0),
        average.get("rouge2", 0),
        average.get("rougeL", 0),
    )

    return result


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def results_to_dataframe(
    results: dict,
    model_name: str = "",
    extra_columns: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Convert per-sample evaluation results into a DataFrame for analysis.

    Args:
        results:       Output from :func:`evaluate_batch`.
        model_name:    Label for the model column.
        extra_columns: Additional columns to add (e.g. preprocessing type).

    Returns:
        DataFrame with one row per sample and columns for each ROUGE metric.
    """
    rows = []
    for i, sample in enumerate(results["per_sample"]):
        if sample is None:
            continue
        row = {"sample_idx": i, "model": model_name}
        for metric, scores in sample.items():
            row[f"{metric}_f1"] = scores["f1"]
            row[f"{metric}_precision"] = scores["precision"]
            row[f"{metric}_recall"] = scores["recall"]
        if extra_columns:
            row.update(extra_columns)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Supplementary metrics
# ---------------------------------------------------------------------------

def compression_ratio(original: str, summary: str) -> float:
    """
    Compute the word-level compression ratio (summary / original).

    A ratio of 0.10 means the summary is 10% of the original length.

    Args:
        original: Source text.
        summary:  Generated summary.

    Returns:
        Compression ratio as a float.
    """
    orig_words = len(original.split()) if original else 0
    summ_words = len(summary.split()) if summary else 0
    return round(summ_words / max(orig_words, 1), 4)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate summary quality")
    parser.add_argument("prediction", help="Path to predicted summary text file")
    parser.add_argument("reference", help="Path to reference summary text file")
    parser.add_argument("--bertscore", action="store_true", help="Include BERTScore")
    args = parser.parse_args()

    from src.utils import read_text

    pred = read_text(args.prediction)
    ref = read_text(args.reference)

    rouge = compute_rouge(pred, ref)
    print("\n=== ROUGE Scores ===")
    print(json.dumps(rouge, indent=2))

    cr = compression_ratio(ref, pred)
    print(f"\nCompression ratio: {cr}")

    if args.bertscore:
        bs = compute_bertscore([pred], [ref])
        print("\n=== BERTScore ===")
        print(json.dumps(bs, indent=2))
