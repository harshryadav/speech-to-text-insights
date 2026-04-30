"""
Tests for the evaluation module.

Run with: pytest tests/test_evaluate.py -v
"""

import pytest

from src.evaluate import (
    compute_rouge,
    compute_rouge_f1,
    compression_ratio,
    evaluate_batch,
)


# ---------------------------------------------------------------------------
# compute_rouge
# ---------------------------------------------------------------------------

class TestComputeRouge:

    def test_perfect_match(self):
        text = "The cat sat on the mat."
        scores = compute_rouge(text, text)
        assert scores["rouge1"]["f1"] == 1.0
        assert scores["rouge2"]["f1"] == 1.0
        assert scores["rougeL"]["f1"] == 1.0

    def test_partial_overlap(self):
        pred = "The cat sat on the mat."
        ref = "The dog sat on the rug."
        scores = compute_rouge(pred, ref)
        assert 0 < scores["rouge1"]["f1"] < 1.0

    def test_no_overlap(self):
        pred = "Hello world."
        ref = "Foo bar baz."
        scores = compute_rouge(pred, ref)
        assert scores["rouge1"]["f1"] == 0.0

    def test_empty_prediction_raises(self):
        with pytest.raises(ValueError, match="Prediction text is empty"):
            compute_rouge("", "reference text")

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="Reference text is empty"):
            compute_rouge("prediction text", "")

    def test_returns_all_metrics(self):
        scores = compute_rouge("The quick fox.", "The lazy fox.")
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores
        for metric in scores.values():
            assert "precision" in metric
            assert "recall" in metric
            assert "f1" in metric

    def test_scores_are_bounded(self):
        scores = compute_rouge("Some text here.", "Other text there.")
        for metric in scores.values():
            for key in ("precision", "recall", "f1"):
                assert 0.0 <= metric[key] <= 1.0


# ---------------------------------------------------------------------------
# compute_rouge_f1 (shorthand)
# ---------------------------------------------------------------------------

class TestComputeRougeF1:

    def test_returns_only_f1(self):
        result = compute_rouge_f1("The cat sat.", "The cat sat.")
        assert set(result.keys()) == {"rouge1", "rouge2", "rougeL"}
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:

    def test_half_length(self):
        original = "one two three four"
        summary = "one two"
        ratio = compression_ratio(original, summary)
        assert ratio == 0.5

    def test_empty_summary(self):
        ratio = compression_ratio("some words here", "")
        assert ratio == 0.0

    def test_empty_original(self):
        ratio = compression_ratio("", "some summary")
        assert ratio > 0  # doesn't crash, returns summary_len / 1


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------

class TestEvaluateBatch:

    def test_batch_of_one(self):
        result = evaluate_batch(
            predictions=["The cat sat."],
            references=["The cat sat."],
        )
        assert result["num_samples"] == 1
        assert result["num_valid"] == 1
        assert result["average"]["rouge1"] == 1.0

    def test_batch_of_multiple(self):
        preds = ["The cat sat.", "Dogs run fast."]
        refs = ["The cat sat.", "Cats run slow."]
        result = evaluate_batch(preds, refs)
        assert result["num_samples"] == 2
        assert result["num_valid"] == 2
        assert len(result["per_sample"]) == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_batch(["a"], ["b", "c"])

    def test_handles_empty_prediction_gracefully(self):
        result = evaluate_batch(
            predictions=["", "The cat sat."],
            references=["A reference.", "The cat sat."],
        )
        # First sample should be skipped (None) due to empty prediction
        assert result["num_valid"] < result["num_samples"]
