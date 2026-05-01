# Speech-to-Text Insights

End-to-end pipeline for automatic speech transcription and summarization, built for DATA/MSML 641: Natural Language Processing.

## Overview

This system converts audio recordings (lectures, meetings, podcasts) into concise text summaries using:

- **Whisper** — Automatic speech recognition (ASR) for audio-to-text transcription
- **TextRank** — Extractive summarization baseline (Mihalcea & Tarau, 2004)
- **BART** — Abstractive summarization via `facebook/bart-large-cnn` (Lewis et al., 2019)
- **T5 / Flan-T5** — Abstractive summarization via `google/flan-t5-base` (Raffel et al., 2020; Chung et al., 2022)
- **Pegasus** — Abstractive summarization via `google/pegasus-cnn_dailymail` (Zhang et al., 2020) — opt-in (~2.2 GB)

The pipeline includes preprocessing (filler word removal, normalization, sentence segmentation), sentence-aware chunking for long transcripts, and evaluation using ROUGE and BERTScore metrics.

## Project Structure

```
├── app.py                         # Streamlit web application
├── configs/config.yaml            # All hyperparameters and model settings
├── requirements.txt               # Python dependencies
│
├── src/
│   ├── utils.py                   # Config loading, logging, I/O, seed
│   ├── transcribe.py              # Whisper ASR wrapper
│   ├── preprocess.py              # Filler removal, normalization, segmentation
│   ├── chunking.py                # Sentence-aware chunking for long inputs
│   ├── summarize_extractive.py    # TextRank baseline
│   ├── summarize_abstractive.py   # BART / T5 summarization
│   ├── evaluate.py                # ROUGE, BERTScore, batch evaluation
│   └── pipeline.py                # End-to-end orchestrator
│
├── tests/                         # Unit tests (pytest)
├── notebooks/                     # Jupyter notebooks for EDA and experiments
├── data/                          # Audio, transcripts, summaries (not in git)
├── results/                       # Evaluation outputs and plots
├── models/                        # Cached model weights (not in git)
└── docs/                          # Detailed project documentation
```

## Setup

### Prerequisites

- Python 3.9+
- pip
- **ffmpeg** (required by Whisper to decode MP3/M4A and other formats — not a Python package)

  **macOS:** `brew install ffmpeg` then verify `ffmpeg -version` in a new terminal.

  **Ubuntu / Debian:** `sudo apt update && sudo apt install -y ffmpeg`

  **Windows:** install from [ffmpeg.org](https://ffmpeg.org/download.html) and add `ffmpeg` to your PATH.

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd FInal_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Verify Installation

```bash
python -c "import whisper; print('Whisper OK')"
python -c "from transformers import pipeline; print('Transformers OK')"
python -c "from src.utils import load_config; print(load_config()); print('Config OK')"
```

## Usage

### Streamlit App (Recommended)

Always run Streamlit with **the same Python** that has your dependencies (your `venv`). If you only type `streamlit run app.py`, macOS may pick a **global** Streamlit and you will see `ModuleNotFoundError: No module named 'whisper'`.

```bash
source venv/bin/activate   # macOS/Linux
python -m streamlit run app.py
```

Or use the launcher (always picks `venv/bin/python`):

```bash
./run_streamlit.sh
```

Then open http://localhost:8501, upload an audio file, and explore the results.

**IDE tip:** In Cursor/VS Code, select the interpreter `FInal_Project/venv/bin/python` so “Run” uses the venv. If the app shows “Interpreter in use: `/usr/local/bin/python3`”, the IDE is still on the wrong Python—use the launcher or **Python: Select Interpreter** first.

### Troubleshooting: `No module named 'whisper'`

1. Confirm Whisper is installed in the venv: `which python` should show `.../FInal_Project/venv/bin/python`, then `python -c "import whisper"`.
2. Start the app with `python -m streamlit run app.py` (not a global `streamlit` on PATH).
3. If the app shows a red box with the interpreter path, that path must be your `venv` Python.

### Hugging Face / VPN (BART & T5)

BART and T5 are downloaded from [huggingface.co](https://huggingface.co) on first use. **Corporate VPNs often block or break HTTPS to that domain**, which shows up as errors like “Can’t load the configuration of `facebook/bart-large-cnn`”.

**Options:**

1. **Use TextRank only** in the app sidebar (works without Hugging Face).
2. **Disconnect VPN** or use a network that allows `huggingface.co`, run the app once so models cache under `~/.cache/huggingface/hub`, then you can go back on VPN (cached files are used offline).
3. **Copy the cache** from another machine: copy `~/.cache/huggingface/` after a successful download.
4. Ask IT to allow traffic to **huggingface.co** (and related CDN hosts if needed).

The Streamlit app skips failed HF models and still shows **Whisper + TextRank** results.

### Command Line — Full Pipeline

```bash
python -m src.pipeline data/raw/meeting.wav \
    --config configs/config.yaml \
    --reference data/reference_summaries/meeting.txt \
    --output results/meeting_results.json
```

### Command Line — Individual Modules

```bash
# Transcribe audio
python -m src.transcribe data/raw/meeting.wav -o data/transcripts/ -m base

# Preprocess transcript
python -m src.preprocess data/transcripts/meeting.json -o data/processed/meeting.json

# Extractive summary
python -m src.summarize_extractive data/processed/meeting_clean.txt -n 5

# Abstractive summary
python -m src.summarize_abstractive data/processed/meeting_clean.txt -m facebook/bart-large-cnn

# Evaluate
python -m src.evaluate results/summary.txt data/reference_summaries/meeting.txt --bertscore
```

### Running Tests

```bash
pytest tests/ -v
```

## Configuration

All hyperparameters live in `configs/config.yaml`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `whisper.model_size` | `base` | Whisper model variant |
| `chunking.max_chunk_tokens` | `800` | Max tokens per chunk |
| `summarization.abstractive.bart.model_name` | `facebook/bart-large-cnn` | BART model |
| `summarization.abstractive.bart.max_summary_length` | `256` | Max summary tokens |
| `summarization.abstractive.t5.model_name` | `google/flan-t5-base` | Flan-T5 model |
| `summarization.abstractive.pegasus.model_name` | `google/pegasus-cnn_dailymail` | Pegasus model (swap to `pegasus-xsum` for one-sentence summaries) |
| `summarization.hierarchical` | `true` | Two-pass summarization for long inputs |

## Documentation

See the `docs/` folder for detailed technical documentation:

- [00-INDEX.md](docs/00-INDEX.md) — Document map and deadlines
- [02-ARCHITECTURE.md](docs/02-ARCHITECTURE.md) — System design and pipeline diagrams
- [06-EVALUATION.md](docs/06-EVALUATION.md) — Metrics and experimental setup
- [07-ENHANCEMENTS.md](docs/07-ENHANCEMENTS.md) — Planned improvements

## References

1. Radford et al. (2022). [Robust Speech Recognition via Large-Scale Weak Supervision.](https://arxiv.org/abs/2212.04356) *arXiv:2212.04356*
2. Lewis et al. (2019). [BART: Denoising Sequence-to-Sequence Pre-training.](https://arxiv.org/abs/1910.13461) *arXiv:1910.13461*
3. Mihalcea & Tarau (2004). [TextRank: Bringing Order into Texts.](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) *EMNLP 2004*
4. Raffel et al. (2020). [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.](https://arxiv.org/abs/1910.10683) *JMLR 21(140)*
5. Zhang et al. (2020). [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.](https://arxiv.org/abs/1912.08777) *ICML 2020*

