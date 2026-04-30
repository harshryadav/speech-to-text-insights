# 04 — Phase 3: Model Development

**Due**: April 22, 2026
**Goal**: BART and T5 abstractive models integrated, ablation studies run, full comparison with TextRank baseline

---

## Deliverables Checklist

- [ ] Advanced model implementations (BART + T5)
- [ ] Hyperparameter tuning and optimization
- [ ] Ablation studies
- [ ] Performance comparison with baselines
- [ ] Error analysis and failure cases
- [ ] Plans for final model and evaluation

---

## 1. BART Integration

### Background

BART (Lewis et al., 2019) is a denoising autoencoder pretrained as a sequence-to-sequence model. The `facebook/bart-large-cnn` variant was fine-tuned on the CNN/DailyMail summarization dataset — making it strong out-of-the-box for summarization without any further fine-tuning.

**Architecture**:
```
Input tokens → BART Encoder (bidirectional) → Latent Representation → BART Decoder (autoregressive) → Summary tokens
```

**Key constraint**: Maximum input length = **1024 tokens**. Meeting transcripts routinely exceed this (a 10-minute meeting can produce 1500-3000 tokens), so chunking is mandatory.

### BART Summarization Flow

```
  Full Transcript (e.g. 2500 tokens)
         │
         ▼
  ┌──────────────────────────────┐
  │  Sentence-Aware Chunking     │
  │  max_tokens=800, overlap=100 │
  └──────────┬───────────────────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  ┌──────┐┌──────┐┌──────┐
  │Chunk1││Chunk2││Chunk3│     Each ≤ 800 tokens
  └──┬───┘└──┬───┘└──┬───┘
     │       │       │
     ▼       ▼       ▼
  ┌──────┐┌──────┐┌──────┐
  │ BART ││ BART ││ BART │     Independent inference
  │ Summ ││ Summ ││ Summ │     max_length=256 per chunk
  └──┬───┘└──┬───┘└──┬───┘
     │       │       │
     ▼       ▼       ▼
  ┌──────────────────────────┐
  │  Concatenate chunk       │
  │  summaries               │
  │  "Summary1. Summary2.    │
  │   Summary3."             │
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  Second-pass summary     │     If concatenation > 1024 tokens
  │  (hierarchical)          │     → run BART again on the
  │                          │       concatenated summaries
  └──────────┬───────────────┘
             │
             ▼
       Final Summary
```

### Implementation: `src/summarize_abstractive.py`

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class AbstractiveSummarizer:
    """Unified interface for BART and T5 summarization."""

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def max_input_length(self) -> int:
        if "bart" in self.model_name.lower():
            return 1024
        elif "t5" in self.model_name.lower():
            return 512
        return 512

    def summarize(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 56,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """Summarize a single chunk of text (must fit within max_input_length)."""
        # T5 requires "summarize: " prefix
        if "t5" in self.model_name.lower():
            text = "summarize: " + text

        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_long(
        self,
        chunks: list[str],
        hierarchical: bool = True,
        max_length: int = 256,
        **kwargs
    ) -> str:
        """
        Summarize a long document via chunking.

        1. Summarize each chunk independently
        2. Concatenate chunk summaries
        3. If hierarchical=True and the concatenation is long,
           run a second summarization pass
        """
        # First pass: summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarize(chunk, max_length=max_length, **kwargs)
            chunk_summaries.append(summary)

        combined = " ".join(chunk_summaries)

        # Second pass (if needed and requested)
        combined_tokens = len(self.tokenizer.encode(combined))
        if hierarchical and combined_tokens > self.max_input_length:
            return self.summarize(combined, max_length=max_length, **kwargs)

        # If combined fits in one pass, optionally refine
        if hierarchical and combined_tokens <= self.max_input_length:
            return self.summarize(combined, max_length=max_length, **kwargs)

        return combined
```

### BART Hyperparameters to Explore

| Parameter | Default | Values to Try | Effect |
|-----------|---------|---------------|--------|
| `max_length` | 256 | 128, 256, 512 | Controls maximum summary length |
| `min_length` | 56 | 30, 56, 80 | Prevents very short outputs |
| `num_beams` | 4 | 2, 4, 6 | Beam search width (higher = better but slower) |
| `length_penalty` | 2.0 | 1.0, 1.5, 2.0, 3.0 | >1 favors longer outputs, <1 favors shorter |
| `no_repeat_ngram_size` | 3 | 2, 3, 4 | Prevents repeating N-grams |

---

## 2. T5 Integration

### Background

T5 (Raffel et al., 2020) treats every NLP task as text-to-text. For summarization, the input is prefixed with `"summarize: "`. We use `t5-base` (220M parameters).

**Key difference from BART**: T5 has a default max input of **512 tokens** (vs BART's 1024), so more chunking is needed.

### T5-Specific Notes

```
Input format:  "summarize: The meeting began with a discussion about..."
                └──prefix──┘ └──────────actual content──────────────┘

Output:        "The meeting covered quarterly results and..."
```

The `AbstractiveSummarizer` class above handles T5 transparently — it detects `"t5"` in the model name and adds the prefix automatically.

### Model Variants to Consider

| Model | Parameters | Speed | Quality |
|-------|-----------|-------|---------|
| `t5-small` | 60M | Fast | Lower quality, good for debugging |
| `t5-base` | 220M | Moderate | Good balance |
| `google/flan-t5-base` | 250M | Moderate | Instruction-tuned, often better |

**Recommendation**: Use `t5-base` for the main comparison, and optionally try `flan-t5-base` as an enhancement (see doc 07).

---

## 3. Chunking Strategy

### Implementation: `src/chunking.py`

```python
from transformers import AutoTokenizer


def chunk_by_sentences(
    sentences: list[str],
    max_tokens: int = 800,
    overlap_sentences: int = 2,
    tokenizer_name: str = "facebook/bart-large-cnn"
) -> list[str]:
    """
    Group sentences into chunks that fit within the token limit.

    Strategy:
    - Add sentences sequentially until token limit is reached
    - Start next chunk with `overlap_sentences` from end of previous chunk
    - Never split a sentence across chunks

    Why sentence-level: Preserves semantic coherence within chunks,
    leading to better chunk-level summaries.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_token_count + sentence_tokens > max_tokens and current_chunk_sentences:
            # Save current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # Start new chunk with overlap
            if overlap_sentences > 0:
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_token_count = len(
                    tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False)
                )
            else:
                current_chunk_sentences = []
                current_token_count = 0

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks
```

### Chunking Experiment Matrix

| Config | max_tokens | overlap | Target model |
|--------|-----------|---------|-------------|
| chunk-bart-800 | 800 | 2 sentences | BART (1024 limit, leave room for special tokens) |
| chunk-bart-512 | 512 | 2 sentences | BART (compare with smaller chunks) |
| chunk-t5-400 | 400 | 2 sentences | T5 (512 limit) |
| chunk-t5-256 | 256 | 2 sentences | T5 (compare with smaller chunks) |

---

## 4. Ablation Studies

The grading rubric explicitly rewards ablation studies (under "Evaluation & Analysis" — 25 points). Run these systematically:

### Ablation Matrix

```
┌───────────────────────────────────────────────────────────────────┐
│                       ABLATION EXPERIMENTS                        │
├───────────────────────┬──────────┬──────────┬──────────┬─────────┤
│ Experiment            │ Variable │ Control  │ Variants │ Metric  │
├───────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ A1: Preprocessing     │ Cleaning │ Raw      │ Cleaned  │ ROUGE   │
│     impact            │          │ transcr. │ transcr. │         │
├───────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ A2: Model comparison  │ Model    │ TextRank │ BART, T5 │ ROUGE   │
│                       │          │          │          │ BERTSc  │
├───────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ A3: Whisper model     │ ASR size │ base     │ small,   │ ROUGE   │
│     size impact       │          │          │ medium   │ WER     │
├───────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ A4: Chunk size        │ Tokens   │ 800      │ 256,512  │ ROUGE   │
│                       │          │          │          │         │
├───────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ A5: Hierarchical vs   │ Strategy │ Single   │ Hierarch │ ROUGE   │
│     single-pass       │          │ pass     │ (2-pass) │ Length  │
├───────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ A6: Beam search width │ Beams    │ 4        │ 1,2,6,8  │ ROUGE   │
│                       │          │          │          │ Speed   │
├───────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ A7: Summary length    │ max_len  │ 256      │ 128, 512 │ ROUGE   │
│                       │          │          │          │ Human   │
└───────────────────────┴──────────┴──────────┴──────────┴─────────┘
```

### How to Run Ablations Efficiently

```python
# Pseudocode for ablation runner
import itertools

configs = {
    "model": ["textrank", "bart", "t5"],
    "preprocessing": ["raw", "cleaned"],
    "chunk_size": [256, 512, 800],
    "hierarchical": [True, False],
}

results = []
for model, preproc, chunk, hier in itertools.product(*configs.values()):
    # Skip invalid combos (textrank doesn't use chunking)
    if model == "textrank" and (chunk != 800 or hier):
        continue

    summary = run_experiment(model, preproc, chunk, hier)
    rouge = compute_rouge(summary, reference)

    results.append({
        "model": model,
        "preprocessing": preproc,
        "chunk_size": chunk,
        "hierarchical": hier,
        **rouge
    })

# Save as DataFrame for analysis
df = pd.DataFrame(results)
df.to_csv("results/ablation_results.csv", index=False)
```

### Expected Results Table (to be filled in)

| Model | Preproc | Chunk | Hier | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|-------|------|---------|---------|---------|
| TextRank | raw | — | — | | | |
| TextRank | clean | — | — | | | |
| BART | raw | 800 | No | | | |
| BART | clean | 800 | No | | | |
| BART | clean | 800 | Yes | | | |
| BART | clean | 512 | Yes | | | |
| T5 | raw | 400 | No | | | |
| T5 | clean | 400 | No | | | |
| T5 | clean | 400 | Yes | | | |
| T5 | clean | 256 | Yes | | | |

---

## 5. Error Analysis Framework

For each model, categorize errors into these types:

### Error Taxonomy

| Error Type | Description | Example |
|-----------|-------------|---------|
| **Hallucination** | Summary contains facts not in transcript | "Revenue grew 15%" (never mentioned) |
| **Omission** | Key topic/decision missing from summary | Budget discussion skipped entirely |
| **Redundancy** | Same point repeated in summary | Same sentence paraphrased twice |
| **Incoherence** | Summary sentences don't flow logically | Abrupt topic shifts within summary |
| **ASR Propagation** | Whisper error causes summary error | "buy/by" confusion changes meaning |
| **Truncation Artifact** | Chunking boundary causes info loss | Mid-topic split loses context |

### Error Analysis Procedure

1. Select 20 test samples (stratified by quality: 5 best, 10 median, 5 worst by ROUGE)
2. For each sample, read the transcript, reference summary, and generated summary
3. Label each error type present (can have multiple per sample)
4. Count frequency of each error type per model
5. Provide 3-5 concrete examples in the report

---

## 6. Day-by-Day Execution Plan (Apr 8 → Apr 22)

| Day | Date | Tasks | Owner |
|-----|------|-------|-------|
| 1 | Apr 8 (Tue) | Implement `chunking.py` with tests | Person A |
| 1 | Apr 8 (Tue) | Start `AbstractiveSummarizer` class | Person B |
| 2 | Apr 9 (Wed) | BART integration: load model, test on 3 samples | Person B |
| 2 | Apr 9 (Wed) | T5 integration: add T5 support to class | Person A |
| 3 | Apr 10 (Thu) | Run BART on full test set (Colab GPU) | Person B |
| 3 | Apr 10 (Thu) | Run T5 on full test set (Colab GPU) | Person A |
| 4 | Apr 11 (Fri) | Implement hierarchical summarization | Both |
| 5 | Apr 12 (Sat) | Run ablation A1 (preprocessing impact) | Person A |
| 5 | Apr 12 (Sat) | Run ablation A2 (model comparison) | Person B |
| 6 | Apr 13 (Sun) | Run ablation A3 (Whisper model size) | Person A |
| 6 | Apr 13 (Sun) | Run ablation A4 (chunk size) | Person B |
| 7 | Apr 14 (Mon) | Run ablations A5-A7 | Both |
| 8 | Apr 15 (Tue) | Compile all results into tables and plots | Person A |
| 9 | Apr 16 (Wed) | Error analysis: review 20 samples per model | Person B |
| 10 | Apr 17 (Thu) | Create results visualization notebook | Person A |
| 10 | Apr 17 (Thu) | Write error analysis section | Person B |
| 11 | Apr 18 (Fri) | Hyperparameter tuning based on ablation insights | Both |
| 12 | Apr 19 (Sat) | Start Streamlit app skeleton | Person A |
| 13 | Apr 20 (Sun) | Write Phase 3 progress report | Both |
| 14 | Apr 21 (Mon) | Review, clean up, submit Phase 3 | Both |
