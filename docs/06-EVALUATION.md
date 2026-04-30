# 06 — Evaluation Strategy

**Purpose**: Define every metric, experimental setup, and statistical test used to measure system performance.

---

## 1. Metrics Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION METRICS                        │
│                                                              │
│  ┌─────────────────────────┐  ┌───────────────────────────┐ │
│  │   AUTOMATIC METRICS     │  │   HUMAN EVALUATION        │ │
│  │                         │  │                           │ │
│  │  • ROUGE-1  (unigram)   │  │  • Clarity      (1-5)    │ │
│  │  • ROUGE-2  (bigram)    │  │  • Coherence    (1-5)    │ │
│  │  • ROUGE-L  (LCS)       │  │  • Relevance    (1-5)    │ │
│  │  • BERTScore (semantic) │  │  • Conciseness  (1-5)    │ │
│  │                         │  │  • Factuality   (1-5)    │ │
│  └─────────────────────────┘  └───────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────┐                                │
│  │   SUPPLEMENTARY         │                                │
│  │                         │                                │
│  │  • Compression ratio    │                                │
│  │  • Processing time      │                                │
│  │  • Word Error Rate      │                                │
│  │    (Whisper only)       │                                │
│  └─────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. ROUGE Scores (Primary Metric)

### What ROUGE Measures

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) compares a generated summary against a human-written reference summary by measuring n-gram overlap.

| Variant | What it measures | Strengths | Weaknesses |
|---------|-----------------|-----------|------------|
| **ROUGE-1** | Unigram overlap | Captures content coverage | Ignores word order |
| **ROUGE-2** | Bigram overlap | Captures some fluency | Sparse for short summaries |
| **ROUGE-L** | Longest Common Subsequence | Captures sentence-level structure | Less sensitive to paraphrasing |

### ROUGE Sub-Scores

Each ROUGE variant produces three numbers:

```
ROUGE-1:
  Precision = (matching unigrams) / (unigrams in generated summary)
  Recall    = (matching unigrams) / (unigrams in reference summary)
  F1        = harmonic mean of Precision and Recall
```

**We report F1 as the primary number** (standard practice). Precision and recall are included in supplementary tables.

### Implementation

```python
from rouge_score import rouge_scorer

def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True  # reduces inflectional variants
    )
    scores = scorer.score(reference, prediction)

    return {
        metric: {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "f1": round(score.fmeasure, 4)
        }
        for metric, score in scores.items()
    }
```

### Interpreting ROUGE Scores

For meeting/lecture summarization (a challenging domain):

| Score Range (F1) | Interpretation |
|-----------------|----------------|
| ROUGE-1 > 0.45 | Excellent |
| ROUGE-1 0.35-0.45 | Good |
| ROUGE-1 0.25-0.35 | Fair (typical extractive baseline) |
| ROUGE-1 < 0.25 | Poor |
| ROUGE-2 > 0.20 | Very good |
| ROUGE-2 0.10-0.20 | Good |
| ROUGE-2 < 0.10 | Typical for extractive |
| ROUGE-L > 0.40 | Very good |
| ROUGE-L 0.30-0.40 | Good |

---

## 3. BERTScore (Semantic Similarity)

### Why BERTScore

ROUGE only measures surface-level overlap. An abstractive summary can be semantically correct but use different words — ROUGE would penalize it. BERTScore uses contextual embeddings to capture semantic similarity.

```
Reference:   "The team decided to increase the marketing budget."
Generated:   "A decision was made to allocate more funds to advertising."

ROUGE-1 F1:   ~0.15  (low — few overlapping words)
BERTScore F1: ~0.82  (high — similar meaning)
```

### Implementation

```python
from bert_score import score as bert_score

def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    P, R, F1 = bert_score(
        predictions,
        references,
        model_type="roberta-large",
        lang="en",
        verbose=False
    )
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4)
    }
```

---

## 4. Human Evaluation

### Protocol

1. **Evaluators**: 2-3 people (team members + 1 external if possible)
2. **Samples**: 20 transcripts from the test set
3. **Blind evaluation**: Evaluators don't know which model generated which summary
4. **Rating scale**: 1-5 Likert scale for each dimension

### Evaluation Form

For each sample, the evaluator sees:
- The original transcript (or a condensed version)
- 3 summaries labeled A, B, C (randomized order: TextRank, BART, T5)

| Dimension | 1 (Worst) | 3 (Average) | 5 (Best) |
|-----------|-----------|-------------|----------|
| **Clarity** | Incomprehensible, grammatically broken | Understandable but awkward | Reads naturally, clear language |
| **Coherence** | Random sentences, no logical flow | Some structure, occasional gaps | Well-organized, logical progression |
| **Relevance** | Misses all key points | Covers some main topics | Captures all important information |
| **Conciseness** | Extremely verbose or too sparse | Reasonable length but some padding | Perfect length, no waste |
| **Factuality** | Contains fabricated facts | Mostly accurate, minor errors | All facts traceable to transcript |

### Inter-Annotator Agreement

Measure agreement using **Cohen's Kappa** (for 2 raters) or **Fleiss' Kappa** (for 3+):

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(rater1_scores, rater2_scores, weights="quadratic")
# Interpretation: >0.8 excellent, 0.6-0.8 good, 0.4-0.6 moderate
```

---

## 5. Supplementary Metrics

### Compression Ratio

```python
def compression_ratio(original: str, summary: str) -> float:
    return len(summary.split()) / len(original.split())

# Ideal range: 0.05 - 0.20 (5-20% of original length)
```

### Processing Time

Track wall-clock time for each pipeline stage:

```python
import time

timings = {}
t0 = time.time()
transcript = transcribe(audio)
timings["transcription"] = time.time() - t0

t0 = time.time()
cleaned = preprocess(transcript)
timings["preprocessing"] = time.time() - t0

t0 = time.time()
summary = summarize(cleaned)
timings["summarization"] = time.time() - t0

timings["total"] = sum(timings.values())
```

### Word Error Rate (for Whisper evaluation)

Only applicable when reference transcripts exist (e.g., AMI corpus has manual transcripts):

```python
# Using the 'jiwer' library
from jiwer import wer

word_error_rate = wer(reference_transcript, whisper_transcript)
# Typical: Whisper-base ~5%, Whisper-medium ~3.5%
```

---

## 6. Experimental Setup

### Data Split

Use the AMI corpus's predefined splits:

| Split | Purpose | Size (approx) |
|-------|---------|---------------|
| Train | Not used (no fine-tuning) | ~80 meetings |
| Validation | Hyperparameter tuning | ~10 meetings |
| Test | Final evaluation | ~10 meetings |

Since we use pretrained models without fine-tuning, the train split is not directly used. We can optionally use a few train samples for development/debugging.

### Experimental Conditions

```
                    ┌──────────────┐
                    │  Full Matrix  │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
     │ TextRank│     │  BART   │     │   T5    │
     └────┬────┘     └────┬────┘     └────┬────┘
          │                │                │
     ┌────┼────┐     ┌────┼────┐     ┌────┼────┐
     │    │    │     │    │    │     │    │    │
    Raw Clean  │    Raw Clean  │    Raw Clean  │
               │               │               │
          (No chunking)   (Chunk variants)  (Chunk variants)
```

Total unique experimental conditions: ~15-20

### Statistical Significance

For comparing two models, use **paired bootstrap resampling**:

```python
import numpy as np

def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000):
    """
    Test whether model A significantly outperforms model B.
    scores_a, scores_b: lists of per-sample ROUGE scores.
    Returns p-value (one-tailed: is A > B?).
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    observed_diff = np.mean(diffs)

    count = 0
    n = len(diffs)
    for _ in range(n_bootstrap):
        sample = np.random.choice(diffs, size=n, replace=True)
        if np.mean(sample) <= 0:
            count += 1

    p_value = count / n_bootstrap
    return observed_diff, p_value

# Usage:
diff, p = paired_bootstrap_test(bart_rouge1_scores, textrank_rouge1_scores)
print(f"BART vs TextRank: Δ={diff:.4f}, p={p:.4f}")
# Report as significant if p < 0.05
```

---

## 7. Results Reporting Templates

### Main Results Table

| Model | Preprocess | ROUGE-1 (F1) | ROUGE-2 (F1) | ROUGE-L (F1) | BERTScore (F1) |
|-------|-----------|--------------|--------------|--------------|----------------|
| TextRank | No | | | | |
| TextRank | Yes | | | | |
| BART | No | | | | |
| BART | Yes | | | | |
| T5 | No | | | | |
| T5 | Yes | | | | |

**Bold** the best score in each column. Include ± standard deviation if computed.

### Ablation Results Table

| Ablation | Variable | Value | ROUGE-1 | ROUGE-2 | ROUGE-L | Δ from default |
|----------|----------|-------|---------|---------|---------|----------------|
| A1 | Preprocessing | Raw | | | | — |
| A1 | Preprocessing | Clean | | | | +X.XX |
| A4 | Chunk size | 256 | | | | |
| A4 | Chunk size | 512 | | | | |
| A4 | Chunk size | 800 | | | | |
| ... | ... | ... | | | | |

### Human Evaluation Table

| Model | Clarity | Coherence | Relevance | Conciseness | Factuality | Avg |
|-------|---------|-----------|-----------|-------------|------------|-----|
| TextRank | | | | | | |
| BART | | | | | | |
| T5 | | | | | | |

### Visualization Suggestions

1. **Grouped bar chart**: ROUGE scores by model (3 groups × 3 bars each)
2. **Box plot**: Score distribution across test samples per model
3. **Radar chart**: Human evaluation dimensions per model
4. **Scatter plot**: ROUGE-1 vs BERTScore (do they correlate?)
5. **Line chart**: Ablation — chunk size vs ROUGE score
6. **Heatmap**: Full ablation matrix (model × preprocessing × chunk size)
