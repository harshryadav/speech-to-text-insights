# 02 — Architecture & Pipelines

## High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SPEECH-TO-TEXT INSIGHTS                          │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────┐ │
│  │  Audio    │──>│  Whisper ASR  │──>│ Preprocessing │──>│ Summary │ │
│  │  Input    │   │ (Transcribe) │   │  (Clean Text) │   │ Engine  │ │
│  └──────────┘   └──────────────┘   └──────────────┘   └────┬────┘ │
│                                                             │      │
│                              ┌───────────────────┬──────────┼──┐   │
│                              │                   │          │  │   │
│                         ┌────▼────┐       ┌──────▼──┐  ┌───▼──▼┐  │
│                         │TextRank │       │  BART    │  │  T5   │  │
│                         │(Extract)│       │(Abstract)│  │(Abstr)│  │
│                         └────┬────┘       └────┬─────┘  └──┬────┘  │
│                              │                 │           │       │
│                              └────────┬────────┘───────────┘       │
│                                       │                            │
│                              ┌────────▼────────┐                   │
│                              │   Evaluation    │                   │
│                              │ (ROUGE, BERTSc) │                   │
│                              └────────┬────────┘                   │
│                                       │                            │
│                              ┌────────▼────────┐                   │
│                              │  Streamlit UI   │                   │
│                              │   (Display)     │                   │
│                              └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Data Flow

### Stage 1: Audio Capture & Transcription

```
                        ┌─────────────────────────────┐
                        │       Audio Sources          │
                        │                              │
                        │  ┌────────┐  ┌────────────┐ │
                        │  │ .wav   │  │ .mp3/.m4a  │ │
                        │  │ files  │  │ files      │ │
                        │  └───┬────┘  └─────┬──────┘ │
                        └──────┼─────────────┼────────┘
                               │             │
                               ▼             ▼
                        ┌─────────────────────────────┐
                        │   librosa / pydub            │
                        │   Audio Loading & Resampling │
                        │   → 16kHz mono WAV           │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │      Whisper Model           │
                        │                              │
                        │  Input:  16kHz audio tensor  │
                        │  Output: {                   │
                        │    "text": "full transcript", │
                        │    "segments": [             │
                        │      {                       │
                        │        "start": 0.0,         │
                        │        "end": 4.2,           │
                        │        "text": "segment..."  │
                        │      }, ...                  │
                        │    ]                         │
                        │  }                           │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │  data/transcripts/           │
                        │  └── meeting_001.json        │
                        │  └── meeting_002.json        │
                        └─────────────────────────────┘
```

**Module**: `src/transcribe.py`

**Key function signatures**:
```python
def transcribe_audio(audio_path: str, model_size: str = "base") -> dict:
    """
    Returns: {
        "text": str,              # full transcript
        "segments": list[dict],   # timestamped segments
        "language": str,          # detected language
        "duration": float         # audio duration in seconds
    }
    """

def batch_transcribe(audio_dir: str, output_dir: str, model_size: str = "base") -> list[str]:
    """Transcribe all audio files in a directory. Returns list of output paths."""
```

### Stage 2: Preprocessing

```
     Raw Transcript
     "Um so basically what we're going to, uh, talk about
      today is the, you know, the quarterly results..."
                               │
                               ▼
              ┌────────────────────────────────┐
              │    Step 1: Filler Removal       │
              │                                 │
              │  Remove: um, uh, like, you know,│
              │  I mean, sort of, basically,    │
              │  actually, right (as fillers)   │
              │                                 │
              │  Regex: r'\b(um|uh|...)\b'      │
              │  + repeated word removal        │
              │  + false start removal          │
              └────────────────┬───────────────┘
                               │
     "So what we're going to talk about today is the
      quarterly results..."
                               │
                               ▼
              ┌────────────────────────────────┐
              │    Step 2: Text Normalization   │
              │                                 │
              │  • Collapse multiple spaces     │
              │  • Fix broken punctuation       │
              │  • Normalize quotes/dashes      │
              │  • Expand common contractions   │
              │    (optional)                   │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │    Step 3: Sentence             │
              │    Segmentation                 │
              │                                 │
              │  spaCy or NLTK sent_tokenize   │
              │  Split into clean sentence list │
              │                                 │
              │  Output: [                     │
              │    "So what we're going...",    │
              │    "The quarterly results...",  │
              │    ...                          │
              │  ]                             │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  data/processed/                │
              │  └── meeting_001_clean.txt      │
              │  └── meeting_001_sentences.json │
              └────────────────────────────────┘
```

**Module**: `src/preprocess.py`

**Key function signatures**:
```python
def remove_fillers(text: str, filler_words: list[str] = None) -> str:
    """Remove filler words and false starts from transcript text."""

def normalize_text(text: str) -> str:
    """Whitespace normalization, punctuation cleanup, encoding fixes."""

def segment_sentences(text: str, method: str = "spacy") -> list[str]:
    """Split text into sentences using spaCy or NLTK."""

def preprocess_transcript(text: str, config: dict) -> dict:
    """
    Full preprocessing pipeline.
    Returns: {
        "original": str,
        "cleaned": str,
        "sentences": list[str],
        "stats": {
            "original_word_count": int,
            "cleaned_word_count": int,
            "fillers_removed": int,
            "sentence_count": int
        }
    }
    """
```

### Stage 3: Chunking (for long transcripts)

```
  Full Cleaned Transcript (e.g. 3000 tokens)
  ┌──────────────────────────────────────────────────┐
  │ Sentence 1. Sentence 2. Sentence 3. Sentence 4. │
  │ Sentence 5. Sentence 6. Sentence 7. Sentence 8. │
  │ Sentence 9. Sentence 10. ... Sentence 50.        │
  └──────────────────────────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │  Sentence-Aware Chunking  │
           │                           │
           │  max_tokens = 800         │
           │  overlap_tokens = 100     │
           │                           │
           │  Rule: Never split mid-   │
           │  sentence. Add complete   │
           │  sentences until limit    │
           │  is reached.             │
           └──────────────┬───────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
     ┌──────────┐  ┌──────────┐  ┌──────────┐
     │ Chunk 1  │  │ Chunk 2  │  │ Chunk 3  │
     │ ~800 tok │  │ ~800 tok │  │ ~400 tok │
     │ Sent 1-18│  │Sent 16-34│  │Sent 32-50│
     └──────────┘  └──────────┘  └──────────┘
      (overlap)─────►(overlap)─────►
```

**Module**: `src/chunking.py`

**Key function signatures**:
```python
def chunk_by_sentences(
    sentences: list[str],
    max_tokens: int = 800,
    overlap_tokens: int = 100,
    tokenizer = None
) -> list[str]:
    """
    Split sentences into chunks respecting token limits.
    Uses overlap for context continuity between chunks.
    """

def chunk_by_tokens(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
    tokenizer = None
) -> list[str]:
    """Token-level chunking (less preferred, may split mid-sentence)."""
```

### Stage 4: Summarization

```
                    Chunked Transcript
                    ┌────┐ ┌────┐ ┌────┐
                    │ C1 │ │ C2 │ │ C3 │
                    └──┬─┘ └──┬─┘ └──┬─┘
                       │      │      │
          ┌────────────┴──────┴──────┴────────────┐
          │                                        │
          ▼                                        ▼
   ┌──────────────┐                      ┌──────────────────┐
   │  EXTRACTIVE   │                      │   ABSTRACTIVE    │
   │  (TextRank)   │                      │  (BART or T5)    │
   │               │                      │                  │
   │  Graph-based  │                      │  Encoder-Decoder │
   │  sentence     │                      │  transformer     │
   │  ranking      │                      │                  │
   │               │                      │  Per-chunk:      │
   │  Selects top  │                      │  ┌────┐ ┌────┐  │
   │  N sentences  │                      │  │ S1 │ │ S2 │  │
   │  from full    │                      │  └──┬─┘ └──┬─┘  │
   │  transcript   │                      │     │      │    │
   │               │                      │     ▼      ▼    │
   │               │                      │  ┌────────────┐ │
   │               │                      │  │ Concatenate │ │
   │               │                      │  └─────┬──────┘ │
   │               │                      │        │        │
   │               │                      │        ▼        │
   │               │                      │  ┌────────────┐ │
   │               │                      │  │ 2nd Pass   │ │
   │               │                      │  │ Summarize  │ │
   │               │                      │  │ (optional) │ │
   │               │                      │  └────────────┘ │
   └──────┬───────┘                      └────────┬─────────┘
          │                                        │
          ▼                                        ▼
   ┌──────────────────────────────────────────────────────┐
   │                   Final Summary                      │
   └──────────────────────────────────────────────────────┘
```

**Module**: `src/summarize_extractive.py`

```python
def textrank_summarize(
    text: str,
    num_sentences: int = 5,
    language: str = "english"
) -> str:
    """
    TextRank (Mihalcea & Tarau, 2004):
    1. Build sentence similarity graph (cosine similarity of TF-IDF vectors)
    2. Run PageRank on the graph
    3. Select top-ranked sentences
    4. Return sentences in original order
    """
```

**Module**: `src/summarize_abstractive.py`

```python
def bart_summarize(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 256,
    min_length: int = 56,
    **kwargs
) -> str:
    """Single-chunk BART summarization."""

def t5_summarize(
    text: str,
    model_name: str = "t5-base",
    max_length: int = 256,
    min_length: int = 56,
    **kwargs
) -> str:
    """Single-chunk T5 summarization. Prepends 'summarize: ' prefix."""

def hierarchical_summarize(
    chunks: list[str],
    model_fn: callable,
    second_pass: bool = True,
    **kwargs
) -> str:
    """
    1. Summarize each chunk independently
    2. Concatenate chunk summaries
    3. If second_pass=True and concatenation exceeds token limit,
       run another summarization pass
    4. Return final summary
    """
```

### Stage 5: Evaluation

```
  ┌──────────────┐      ┌──────────────────┐
  │  Generated   │      │  Reference       │
  │  Summary     │      │  Summary         │
  │  (Predicted) │      │  (Gold Standard) │
  └──────┬───────┘      └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │    Automatic Metrics  │
         │                       │
         │  ┌─────────────────┐  │
         │  │ ROUGE-1 (uni)   │  │  ← unigram overlap
         │  │ ROUGE-2 (bi)    │  │  ← bigram overlap
         │  │ ROUGE-L (LCS)   │  │  ← longest common subsequence
         │  └─────────────────┘  │
         │                       │
         │  ┌─────────────────┐  │
         │  │ BERTScore       │  │  ← semantic similarity via embeddings
         │  └─────────────────┘  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Human Evaluation    │
         │                       │
         │  Clarity     (1-5)    │
         │  Coherence   (1-5)    │
         │  Relevance   (1-5)    │
         │  Conciseness (1-5)    │
         └───────────────────────┘
```

### Stage 6: Streamlit Application

```
  ┌─────────────────────────────────────────────────────────┐
  │                    Streamlit App                         │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Sidebar                                         │   │
  │  │  ┌───────────────────────┐                      │   │
  │  │  │ Upload Audio File     │  (.wav, .mp3, .m4a)  │   │
  │  │  └───────────────────────┘                      │   │
  │  │  ┌───────────────────────┐                      │   │
  │  │  │ Whisper Model Size    │  [base ▼]            │   │
  │  │  └───────────────────────┘                      │   │
  │  │  ┌───────────────────────┐                      │   │
  │  │  │ Summarization Method  │  [All / BART / T5]   │   │
  │  │  └───────────────────────┘                      │   │
  │  │  ┌───────────────────────┐                      │   │
  │  │  │ [▶ Process]           │                      │   │
  │  │  └───────────────────────┘                      │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Tab 1: Transcript                               │   │
  │  │  ┌─────────────────┬───────────────────────┐    │   │
  │  │  │ Raw Transcript  │ Cleaned Transcript     │    │   │
  │  │  │ (with fillers)  │ (fillers removed)      │    │   │
  │  │  └─────────────────┴───────────────────────┘    │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Tab 2: Summaries                                │   │
  │  │  ┌──────────┬──────────┬──────────┐             │   │
  │  │  │ TextRank │  BART    │   T5     │             │   │
  │  │  │          │          │          │             │   │
  │  │  │ (extr.)  │ (abstr.) │ (abstr.) │             │   │
  │  │  └──────────┴──────────┴──────────┘             │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Tab 3: Evaluation (if reference provided)       │   │
  │  │  ROUGE scores table + bar chart                  │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Tab 4: Statistics                               │   │
  │  │  Word counts, compression ratio, time taken      │   │
  │  └─────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
                    app.py (Streamlit)
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
    transcribe.py  preprocess.py  evaluate.py
          │             │             │
          │             ▼             │
          │       chunking.py        │
          │             │             │
          │      ┌──────┴──────┐     │
          │      ▼             ▼     │
          │ summarize_     summarize_ │
          │ extractive.py  abstractive.py
          │      │             │     │
          └──────┴──────┬──────┴─────┘
                        │
                        ▼
                    utils.py
                        │
                        ▼
                  configs/config.yaml
```

## Data Flow Summary

```
Audio File (.wav/.mp3)
    │
    ├─ transcribe.py ──────────► Raw Transcript (JSON)
    │                                   │
    ├─ preprocess.py ──────────► Cleaned Transcript (TXT + JSON)
    │                                   │
    ├─ chunking.py ────────────► List of Chunks
    │                                   │
    ├─ summarize_extractive.py ► TextRank Summary
    ├─ summarize_abstractive.py ► BART Summary
    ├─ summarize_abstractive.py ► T5 Summary
    │                                   │
    ├─ evaluate.py ────────────► ROUGE / BERTScore Results
    │                                   │
    └─ app.py ─────────────────► Interactive Dashboard
```
