# 05 — Phase 4: Final Deliverables

**Due**: May 13, 2026
**Deliverables**: Technical Report (15-20 pages) + Code Repository + Presentation (15 min + 5 min Q&A)

---

## Deliverables Checklist

- [ ] Technical report (15-20 pages, structured per guidelines)
- [ ] Clean, well-documented code repository
- [ ] README with setup and usage instructions
- [ ] Reproducible results with random seeds
- [ ] Unit tests for key functions
- [ ] Streamlit demo application
- [ ] Presentation slides (15 min)
- [ ] Live demo prepared

---

## 1. Technical Report

### Structure (following course guidelines exactly)

```
┌─────────────────────────────────────────┐
│           TECHNICAL REPORT              │
│           (15-20 pages)                 │
├─────────────────────────────────────────┤
│                                         │
│  1. Abstract                (½ page)    │
│     - Problem, approach, key result     │
│                                         │
│  2. Introduction            (2-3 pages) │
│     - Problem motivation                │
│     - Why audio summarization matters   │
│     - Contribution summary              │
│     - Paper organization                │
│                                         │
│  3. Related Work            (2-3 pages) │
│     - ASR: Whisper and alternatives     │
│     - Extractive: TextRank, LexRank     │
│     - Abstractive: BART, T5, Pegasus    │
│     - Meeting summarization systems     │
│     - How our work differs              │
│                                         │
│  4. Methodology             (4-5 pages) │
│     - System architecture diagram       │
│     - Data collection (AMI corpus)      │
│     - Preprocessing pipeline            │
│     - Chunking strategy                 │
│     - Model descriptions                │
│       - TextRank                        │
│       - BART                            │
│       - T5                              │
│     - Hyperparameters table             │
│                                         │
│  5. Experiments             (3-4 pages) │
│     - Experimental setup                │
│     - Evaluation metrics                │
│     - Main results table                │
│     - Ablation study results            │
│     - Statistical significance          │
│                                         │
│  6. Analysis                (2-3 pages) │
│     - Error analysis with examples      │
│     - Failure cases                     │
│     - Ablation insights                 │
│     - Computational cost analysis       │
│                                         │
│  7. Discussion              (1-2 pages) │
│     - Limitations                       │
│     - Ethical considerations            │
│     - Real-world deployment notes       │
│     - Future work                       │
│                                         │
│  8. Conclusion              (1 page)    │
│                                         │
│  9. References              (2+ pages)  │
│     - 15+ papers                        │
│                                         │
└─────────────────────────────────────────┘
```

### Reference List (starter — expand to 15+)

Core references (from your proposal):
1. Radford et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." arXiv:2212.04356
2. Lewis et al. (2019). "BART: Denoising Sequence-to-Sequence Pre-training." arXiv:1910.13461
3. Mihalcea & Tarau (2004). "TextRank: Bringing Order into Texts." EMNLP 2004.

Additional references to include:
4. Raffel et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." (T5) JMLR.
5. Lin (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." ACL Workshop.
6. Zhang et al. (2020). "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.
7. Carletta et al. (2005). "The AMI Meeting Corpus." MLMI Workshop. (dataset paper)
8. Erber et al. (2010). "The ICSI Meeting Corpus." (secondary dataset)
9. Nallapati et al. (2017). "SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization." AAAI.
10. See et al. (2017). "Get To The Point: Summarization with Pointer-Generator Networks." ACL.
11. Zhang et al. (2020). "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization." ICML.
12. Zhong et al. (2021). "QMSum: A New Benchmark for Query-Based Multi-Domain Meeting Summarization." NAACL.
13. Vaswani et al. (2017). "Attention Is All You Need." NeurIPS. (Transformer architecture)
14. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
15. Liu & Lapata (2019). "Text Summarization with Pretrained Encoders." EMNLP.

### Writing Tips for High Scores

- **Figures**: Include at least 4-5 figures (architecture diagram, results charts, error examples)
- **Tables**: At least 3 tables (main results, ablation results, hyperparameters)
- **Quantitative claims**: Always back with numbers ("BART improved ROUGE-1 by 12.3 points over TextRank")
- **Honest limitations**: The rubric rewards acknowledging what didn't work
- **Formatting**: Use LaTeX or clean Markdown → PDF. Consistent fonts, numbered sections

---

## 2. Code Repository Requirements

### README.md Template

```markdown
# Speech-to-Text Insights

End-to-end pipeline for automatic speech transcription and summarization.

## Overview

This system converts audio recordings (lectures, meetings, podcasts) into
concise text summaries using:
- **Whisper** for automatic speech recognition
- **TextRank** for extractive summarization (baseline)
- **BART** and **T5** for abstractive summarization

## Setup

### Prerequisites
- Python 3.9+
- pip

### Installation
git clone <repo-url>
cd FInal_Project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

### Quick Start

# Transcribe an audio file
python -m src.transcribe --input audio.wav --output transcript.json

# Preprocess transcript
python -m src.preprocess --input transcript.json --output cleaned.json

# Summarize
python -m src.summarize_abstractive --input cleaned.json --model bart --output summary.txt

# Run Streamlit app
streamlit run app.py

## Project Structure
(include tree from doc 01)

## Evaluation
python -m src.evaluate --predictions results/summaries/ --references data/reference_summaries/

## Results
(summary table of key results)

## Team
- Person A: [Role]
- Person B: [Role]

## License
MIT
```

### Code Quality Checklist

- [ ] Every `.py` file has module-level docstring
- [ ] Every public function has docstring with Args/Returns
- [ ] Type hints on all function signatures
- [ ] Config values loaded from `config.yaml`, not hardcoded
- [ ] Random seeds set for reproducibility
- [ ] No API keys or secrets in code
- [ ] `__main__` blocks for running modules standalone
- [ ] Logging instead of print statements
- [ ] Error handling for file I/O and model loading

### Unit Tests

```python
# tests/test_preprocess.py
def test_remove_fillers_basic():
    text = "Um so basically we need to, uh, discuss this"
    cleaned, count = remove_fillers(text)
    assert "um" not in cleaned.lower()
    assert "uh" not in cleaned.lower()
    assert count >= 3

def test_normalize_text_whitespace():
    text = "Hello   world  .  This is   a test ."
    result = normalize_text(text)
    assert "  " not in result
    assert result.endswith(".")

def test_segment_sentences():
    text = "First sentence. Second sentence. Third one."
    sentences = segment_sentences(text)
    assert len(sentences) == 3

# tests/test_chunking.py
def test_chunking_respects_limit():
    sentences = ["Sentence " * 50 + "." for _ in range(20)]
    chunks = chunk_by_sentences(sentences, max_tokens=200)
    # Verify each chunk is within limit (approximately)
    for chunk in chunks:
        assert len(chunk.split()) < 250  # rough check

def test_chunking_no_empty_chunks():
    sentences = ["Short.", "Also short."]
    chunks = chunk_by_sentences(sentences, max_tokens=1000)
    assert all(len(c.strip()) > 0 for c in chunks)
```

### Reproducibility

```python
# In every script that uses randomness:
import random
import numpy as np
import torch

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## 3. Streamlit Application

### Full App Structure

```python
# app.py — Top-level Streamlit application

import streamlit as st

st.set_page_config(
    page_title="Speech-to-Text Insights",
    page_icon="🎙️",
    layout="wide"
)

# --- Sidebar ---
# File upload
# Model selection
# Parameter controls
# Process button

# --- Main Area ---
# Tab 1: Transcript (raw vs cleaned, side by side)
# Tab 2: Summaries (TextRank vs BART vs T5, side by side)
# Tab 3: Evaluation (ROUGE table + charts, if reference provided)
# Tab 4: Statistics (word counts, compression ratios, timing)

# --- Footer ---
# Project info, team, course
```

### Key UI Features

1. **Audio upload**: Accept `.wav`, `.mp3`, `.m4a` via `st.file_uploader`
2. **Progress indicators**: `st.progress()` and `st.spinner()` during Whisper/BART processing
3. **Side-by-side comparison**: `st.columns(3)` for three summaries
4. **Metrics display**: `st.metric()` cards for ROUGE scores
5. **Charts**: `st.plotly_chart()` for ROUGE comparison bar charts
6. **Download**: `st.download_button()` for transcript and summary files
7. **Optional reference upload**: For evaluation mode, let user upload a reference summary

### Caching Strategy

```python
@st.cache_resource
def load_whisper_model(size):
    return whisper.load_model(size)

@st.cache_resource
def load_summarizer(model_name):
    return AbstractiveSummarizer(model_name)

@st.cache_data
def transcribe_cached(audio_bytes, model_size):
    # Save to temp file, transcribe, return result
    ...
```

---

## 4. Presentation

### Slide Outline (15 minutes)

| Slide | Content | Time | Notes |
|-------|---------|------|-------|
| 1 | Title slide | 0:30 | Project name, team, course |
| 2 | Problem definition | 1:00 | Why manual listening is inefficient |
| 3 | Motivation & use cases | 1:00 | Lectures, meetings, podcasts, accessibility |
| 4 | Related work | 1:30 | Whisper, BART, T5, TextRank — 1 sentence each |
| 5 | System architecture | 2:00 | Pipeline diagram — walk through each stage |
| 6 | Data & preprocessing | 1:30 | AMI corpus, filler removal examples |
| 7 | Models overview | 2:00 | TextRank vs BART vs T5 — key differences |
| 8 | Results | 2:00 | Main ROUGE table, best model highlighted |
| 9 | Ablation highlights | 1:00 | Most interesting findings (preprocessing impact, etc.) |
| 10 | Live demo | 2:00 | Streamlit app: upload audio → see summary |
| 11 | Error analysis | 1:00 | 2-3 examples of failures and why |
| 12 | Conclusion & future work | 1:00 | Summary, limitations, next steps |

### Demo Preparation

- Pre-load 2-3 audio samples of varying length (30s, 2min, 5min)
- Have pre-computed results cached in case of live failure
- Test on presentation laptop beforehand
- Have screenshots as backup slides if Streamlit crashes

---

## 5. Day-by-Day Execution Plan (Apr 22 → May 13)

| Day | Date | Tasks | Owner |
|-----|------|-------|-------|
| 1-3 | Apr 22-24 | Complete Streamlit app (all tabs working) | Person A |
| 1-3 | Apr 22-24 | Start report: Introduction + Related Work | Person B |
| 4-5 | Apr 25-26 | Add evaluation tab + charts to Streamlit | Person A |
| 4-5 | Apr 25-26 | Report: Methodology section | Person B |
| 6-7 | Apr 27-28 | Final experiment runs, fill all result tables | Both |
| 8-9 | Apr 29-30 | Report: Experiments + Analysis sections | Both |
| 10 | May 1 (Thu) | Human evaluation: 3 evaluators rate 20 samples | Both |
| 11-12 | May 2-3 | Report: Discussion + Conclusion | Person B |
| 11-12 | May 2-3 | Write unit tests, clean code, add docstrings | Person A |
| 13-14 | May 4-5 | Report: Abstract, References, formatting polish | Both |
| 15 | May 6 (Tue) | Create presentation slides | Both |
| 16 | May 7 (Wed) | Rehearse presentation, time it | Both |
| 17 | May 8 (Thu) | Prepare demo, test on presentation machine | Person A |
| 18 | May 9 (Fri) | Final report review, peer-edit | Both |
| 19 | May 10 (Sat) | Buffer day — fix anything outstanding | Both |
| 20 | May 11 (Sun) | Final README, .gitignore, repo cleanup | Both |
| 21 | May 12 (Mon) | Final rehearsal, submit report + code | Both |
| 22 | May 13 (Tue) | **Presentation day** | Both |
