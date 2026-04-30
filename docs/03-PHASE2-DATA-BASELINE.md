# 03 — Phase 2: Data Collection & Baseline

**Due**: April 8, 2026
**Goal**: Working pipeline from audio → transcript → cleaned text → TextRank summary, with ROUGE scores

---

## Deliverables Checklist

- [ ] Data collection and preprocessing pipeline
- [ ] Exploratory data analysis
- [ ] Baseline model implementation (TextRank)
- [ ] Initial results and error analysis
- [ ] Updated project timeline
- [ ] Challenges encountered and solutions

---

## 1. Dataset Selection & Acquisition

### Primary Dataset: AMI Meeting Corpus

**Why AMI**: It provides real meeting audio with human-written abstractive and extractive summaries — a direct match for our use case.

| Property | Details |
|----------|---------|
| Size | ~100 hours of meetings |
| Format | Audio (WAV) + manual transcripts + summaries |
| Speakers | 3-5 per meeting |
| License | CC BY 4.0 (free for research) |
| Download | [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) |

**Download steps**:
```bash
# Option A: Via HuggingFace datasets (recommended — easiest)
python -c "
from datasets import load_dataset
ds = load_dataset('edinburghcstr/ami', 'ihm', split='test')
print(ds)
print(ds[0].keys())
"

# Option B: Direct download
# Visit: https://groups.inf.ed.ac.uk/ami/download/
# Download the "Individual Headset Mix" (IHM) audio
# Download the abstractive summaries from the annotations
```

**HuggingFace `datasets` approach** (preferred):
```python
from datasets import load_dataset

# Load AMI with individual headset microphone audio
ami = load_dataset("edinburghcstr/ami", "ihm")

# Structure:
# ami['train'], ami['validation'], ami['test']
# Each sample: {
#   'meeting_id': str,
#   'audio': {'array': np.ndarray, 'sampling_rate': int},
#   'text': str  (transcript)
# }

# For summaries, load the summary subset:
ami_summary = load_dataset("edinburghcstr/ami", "ihm", split="test")
```

### Secondary Dataset: Custom Recordings

Record 5-10 short clips (3-10 minutes each) of:
- A mock meeting discussion
- A lecture segment
- A podcast-style conversation
- A presentation with Q&A

These serve as **demo material** for the Streamlit app and test real-world robustness.

**Recording guidelines**:
- Use any phone or laptop microphone
- Mix quiet and noisy environments
- Vary number of speakers (1-3)
- Save as `.wav` or `.mp3` at 16kHz+
- Write a brief reference summary for each (2-4 sentences)

### Optional Supplementary Datasets

| Dataset | Use | Link |
|---------|-----|------|
| ICSI Meeting Corpus | Additional meeting data | [ICSI](https://groups.inf.ed.ac.uk/ami/icsi/) |
| TED-LIUM 3 | Lecture-style audio with transcripts | [HuggingFace](https://huggingface.co/datasets/LIUM/tedlium) |
| LibriSpeech (test-clean) | WER benchmarking for Whisper | [OpenSLR](https://www.openslr.org/12) |

---

## 2. Whisper Transcription Pipeline

### Technical Background

Whisper (Radford et al., 2022) is an encoder-decoder transformer trained on 680K hours of web audio. Key properties:
- Trained on weakly supervised data (noisy internet transcripts)
- Robust to accents, background noise, technical language
- Supports 99 languages (we use English)
- Multiple model sizes with different speed/accuracy tradeoffs

### Model Size Selection

| Model | Parameters | VRAM | Relative Speed | English WER |
|-------|-----------|------|----------------|-------------|
| tiny | 39M | ~1 GB | ~32x | ~8.0% |
| base | 74M | ~1 GB | ~16x | ~5.5% |
| small | 244M | ~2 GB | ~6x | ~4.2% |
| medium | 769M | ~5 GB | ~2x | ~3.5% |
| large-v3 | 1550M | ~10 GB | 1x | ~2.7% |

**Recommendation**: Start with `base` for fast iteration. Use `small` or `medium` for final results.

### Implementation: `src/transcribe.py`

```python
import whisper
import json
import os
import time
from pathlib import Path
from tqdm import tqdm


def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """Load and cache a Whisper model."""
    return whisper.load_model(model_size)


def transcribe_audio(
    audio_path: str,
    model: whisper.Whisper = None,
    model_size: str = "base",
    language: str = "en"
) -> dict:
    """
    Transcribe a single audio file.

    Returns dict with keys: text, segments, language, duration, processing_time
    """
    if model is None:
        model = load_whisper_model(model_size)

    start_time = time.time()

    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        verbose=False
    )

    processing_time = time.time() - start_time

    # Get audio duration
    audio = whisper.load_audio(audio_path)
    duration = len(audio) / whisper.audio.SAMPLE_RATE

    return {
        "audio_file": os.path.basename(audio_path),
        "text": result["text"].strip(),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ],
        "language": result.get("language", language),
        "duration_seconds": round(duration, 2),
        "processing_time_seconds": round(processing_time, 2),
        "model_size": model_size,
        "word_count": len(result["text"].split())
    }


def batch_transcribe(
    audio_dir: str,
    output_dir: str,
    model_size: str = "base",
    extensions: tuple = (".wav", ".mp3", ".m4a", ".flac")
) -> list[str]:
    """Transcribe all audio files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    model = load_whisper_model(model_size)

    audio_files = [
        f for f in Path(audio_dir).iterdir()
        if f.suffix.lower() in extensions
    ]

    output_paths = []
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        result = transcribe_audio(str(audio_file), model=model, model_size=model_size)

        output_path = os.path.join(output_dir, f"{audio_file.stem}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        output_paths.append(output_path)

    return output_paths
```

### Expected Output Format

```json
{
  "audio_file": "meeting_001.wav",
  "text": "Good morning everyone. Today we're going to discuss the quarterly results...",
  "segments": [
    {"start": 0.0, "end": 2.1, "text": "Good morning everyone."},
    {"start": 2.1, "end": 5.8, "text": "Today we're going to discuss the quarterly results."}
  ],
  "language": "en",
  "duration_seconds": 342.5,
  "processing_time_seconds": 28.3,
  "model_size": "base",
  "word_count": 1523
}
```

---

## 3. Preprocessing Pipeline

### Implementation: `src/preprocess.py`

```python
import re
import nltk
from typing import Optional

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


DEFAULT_FILLERS = [
    "um", "uh", "erm", "ah", "eh",
    "like",  # when used as filler (imperfect — context-free removal)
    "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally",
    "right", "okay so", "so yeah"
]


def remove_fillers(text: str, filler_words: list[str] = None) -> tuple[str, int]:
    """
    Remove filler words from text.
    Returns (cleaned_text, count_removed).

    Strategy: Word-boundary regex matching, case-insensitive.
    Limitation: "like" as filler vs "like" as verb — we accept
    some false positives for cleaner output.
    """
    if filler_words is None:
        filler_words = DEFAULT_FILLERS

    count = 0
    result = text

    # Sort by length (longest first) to avoid partial matches
    filler_words_sorted = sorted(filler_words, key=len, reverse=True)

    for filler in filler_words_sorted:
        pattern = r'\b' + re.escape(filler) + r'\b[,]?\s*'
        matches = re.findall(pattern, result, flags=re.IGNORECASE)
        count += len(matches)
        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)

    # Remove repeated words ("the the", "we we")
    result = re.sub(r'\b(\w+)\s+\1\b', r'\1', result, flags=re.IGNORECASE)

    return result.strip(), count


def normalize_text(text: str) -> str:
    """
    Normalize whitespace, punctuation, and encoding artifacts.
    """
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # Normalize quotes and dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', ' -- ').replace('\u2013', ' - ')

    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = ' '.join(line for line in lines if line)

    return text.strip()


def segment_sentences(text: str, method: str = "nltk") -> list[str]:
    """
    Split text into sentences.
    Methods: 'nltk' (fast, good enough) or 'spacy' (slower, better for edge cases).
    """
    if method == "nltk":
        return nltk.sent_tokenize(text)
    elif method == "spacy":
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    else:
        raise ValueError(f"Unknown method: {method}")


def preprocess_transcript(text: str, config: dict = None) -> dict:
    """
    Full preprocessing pipeline: filler removal → normalization → segmentation.
    """
    original_word_count = len(text.split())

    # Step 1: Remove fillers
    filler_words = config.get("filler_words", DEFAULT_FILLERS) if config else DEFAULT_FILLERS
    cleaned, fillers_removed = remove_fillers(text, filler_words)

    # Step 2: Normalize
    cleaned = normalize_text(cleaned)

    # Step 3: Segment
    sentences = segment_sentences(cleaned)

    return {
        "original": text,
        "cleaned": cleaned,
        "sentences": sentences,
        "stats": {
            "original_word_count": original_word_count,
            "cleaned_word_count": len(cleaned.split()),
            "fillers_removed": fillers_removed,
            "sentence_count": len(sentences),
            "compression_ratio": round(len(cleaned.split()) / max(original_word_count, 1), 3)
        }
    }
```

### Preprocessing Quality Checks

After running preprocessing, verify:

1. **No information loss**: Spot-check that meaningful content survives
2. **Filler removal accuracy**: Sample 20 removals — are they actual fillers?
3. **Sentence boundaries**: Are sentences split correctly? Check for over/under-splitting
4. **Word count reduction**: Expect ~5-15% reduction from filler removal

---

## 4. TextRank Baseline

### How TextRank Works (Mihalcea & Tarau, 2004)

```
    Input: List of sentences
              │
              ▼
    ┌─────────────────────────┐
    │ 1. Build similarity     │     Each sentence → TF-IDF vector
    │    matrix between all   │     Similarity = cosine(sent_i, sent_j)
    │    sentence pairs       │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ 2. Construct graph      │     Nodes = sentences
    │                         │     Edges = similarity scores (weighted)
    │    S1 ──0.3── S2        │     Threshold: drop edges below 0.1
    │    │ \        │         │
    │   0.5  0.1   0.4       │
    │    │     \    │         │
    │    S3 ──0.2── S4        │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ 3. Run PageRank         │     Iterative algorithm converges
    │                         │     to steady-state importance scores
    │    S1: 0.32             │
    │    S2: 0.18             │
    │    S3: 0.28             │
    │    S4: 0.22             │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ 4. Select top-N         │     Sorted by score, returned in
    │    sentences            │     ORIGINAL document order
    │                         │
    │    Output: S1, S3       │
    └─────────────────────────┘
```

### Implementation: `src/summarize_extractive.py`

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


def textrank_summarize(
    text: str,
    num_sentences: int = 5,
    language: str = "english"
) -> str:
    """
    Extractive summarization using TextRank.
    Uses the `sumy` library which implements the original algorithm.
    """
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)

    summary_sentences = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary_sentences)


def textrank_summarize_ratio(
    text: str,
    ratio: float = 0.2,
    language: str = "english"
) -> str:
    """Summarize to approximately `ratio` proportion of original sentences."""
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    total_sentences = len(parser.document.sentences)
    num_sentences = max(1, int(total_sentences * ratio))
    return textrank_summarize(text, num_sentences=num_sentences, language=language)
```

### Baseline Experiment Plan

Run TextRank on the test set with these variations:

| Experiment ID | Input | Num Sentences | Notes |
|--------------|-------|---------------|-------|
| TR-raw-5 | Raw transcript | 5 | No preprocessing |
| TR-raw-10 | Raw transcript | 10 | No preprocessing |
| TR-clean-5 | Cleaned transcript | 5 | With preprocessing |
| TR-clean-10 | Cleaned transcript | 10 | With preprocessing |
| TR-clean-ratio | Cleaned transcript | 20% of total | Adaptive length |

This gives us a baseline ROUGE table and also shows **preprocessing impact**.

---

## 5. Evaluation (Initial)

### Implementation: `src/evaluate.py`

```python
from rouge_score import rouge_scorer


def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
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


def evaluate_batch(predictions: list[str], references: list[str]) -> dict:
    """Compute average ROUGE scores across a batch."""
    all_scores = [compute_rouge(p, r) for p, r in zip(predictions, references)]

    avg_scores = {}
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        for measure in ['precision', 'recall', 'f1']:
            values = [s[metric][measure] for s in all_scores]
            avg_scores[f"{metric}_{measure}"] = round(sum(values) / len(values), 4)

    return avg_scores
```

### Expected Baseline Results (approximate)

Based on literature, TextRank on meeting transcripts typically yields:

| Metric | Expected Range |
|--------|---------------|
| ROUGE-1 F1 | 0.25 - 0.35 |
| ROUGE-2 F1 | 0.05 - 0.12 |
| ROUGE-L F1 | 0.20 - 0.30 |

These are modest scores — the abstractive models in Phase 3 should improve significantly.

---

## 6. Exploratory Data Analysis (EDA)

Create `notebooks/01_eda.ipynb` covering:

### Audio Statistics
- Number of recordings in train/val/test splits
- Duration distribution (histogram)
- Audio quality metrics (signal-to-noise ratio if available)

### Transcript Statistics
- Word count distribution
- Token count distribution (using BART tokenizer for relevance)
- Average/median/max transcript length
- How many transcripts exceed 1024 tokens (BART's limit)?

### Preprocessing Impact
- Before vs after word counts
- Types and frequency of fillers removed
- Sentence count distribution after segmentation

### Reference Summary Statistics
- Summary length distribution
- Compression ratio (summary length / transcript length)
- Vocabulary overlap between transcript and summary

---

## 7. Progress Report Template

Use this structure for the Phase 2 submission:

```markdown
# Phase 2 Progress Report — Speech-to-Text Insights

## 1. Data Collection
- Dataset: AMI Meeting Corpus (describe splits used)
- Custom recordings: N files, total duration
- Data format and storage

## 2. Preprocessing Pipeline
- Steps implemented (with examples)
- Statistics: avg fillers removed, compression ratio

## 3. Baseline Implementation
- TextRank configuration
- How chunking is handled (if applicable)

## 4. Initial Results
- ROUGE scores table (TextRank variants)
- Best configuration identified

## 5. Error Analysis
- Example of good summary
- Example of bad summary
- Common failure patterns

## 6. Challenges Encountered
- [List specific challenges and how they were resolved]

## 7. Updated Timeline
- What remains for Phase 3 and Phase 4
```

---

## 8. Day-by-Day Execution Plan (Mar 31 → Apr 8)

| Day | Date | Tasks | Owner |
|-----|------|-------|-------|
| 1 | Mar 31 (Mon) | Set up repo structure, install deps, create .gitignore | Both |
| 2 | Apr 1 (Tue) | Download AMI corpus via HuggingFace, explore data structure | Person A |
| 2 | Apr 1 (Tue) | Implement `preprocess.py` with tests | Person B |
| 3 | Apr 2 (Wed) | Implement `transcribe.py`, run Whisper on 5-10 AMI samples | Person A |
| 3 | Apr 2 (Wed) | Implement `summarize_extractive.py` (TextRank) | Person B |
| 4 | Apr 3 (Thu) | Run transcription on full test set (Colab GPU) | Person A |
| 4 | Apr 3 (Thu) | Implement `evaluate.py`, run TextRank baseline | Person B |
| 5 | Apr 4 (Fri) | Run all baseline experiments (TR-raw vs TR-clean) | Both |
| 6 | Apr 5 (Sat) | Create EDA notebook, generate statistics and plots | Person A |
| 6 | Apr 5 (Sat) | Error analysis: review 10 best and 10 worst summaries | Person B |
| 7 | Apr 6 (Sun) | Write progress report, clean up code, add docstrings | Both |
| 8 | Apr 7 (Mon) | Final review, submit Phase 2 | Both |
