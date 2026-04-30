# 01 — Project Setup & Environment

## Repository Structure

```
FInal_Project/
│
├── README.md                      # Project overview, setup instructions, usage
├── requirements.txt               # Python dependencies with versions
├── .gitignore                     # Ignore data/, models/, __pycache__, etc.
├── guidelines.md                  # Course guidelines (existing)
├── docs/                          # This documentation folder
│
├── src/
│   ├── __init__.py
│   ├── transcribe.py              # Whisper ASR wrapper
│   ├── preprocess.py              # Filler removal, segmentation, normalization
│   ├── summarize_extractive.py    # TextRank baseline
│   ├── summarize_abstractive.py   # BART / T5 summarization
│   ├── chunking.py                # Transcript chunking for long inputs
│   ├── evaluate.py                # ROUGE scoring, BERTScore, human eval
│   └── utils.py                   # Shared I/O, logging, config loading
│
├── app.py                         # Streamlit application entry point
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_whisper_experiments.ipynb
│   ├── 03_summarization_comparison.ipynb
│   └── 04_results_visualization.ipynb
│
├── configs/
│   └── config.yaml                # All hyperparameters and model settings
│
├── data/
│   ├── raw/                       # Original audio files (NOT committed to git)
│   ├── transcripts/               # Whisper JSON outputs
│   ├── processed/                 # Cleaned transcripts
│   ├── reference_summaries/       # Gold-standard summaries for evaluation
│   └── samples/                   # Small sample files for testing / demo
│
├── results/
│   ├── rouge_scores/              # CSV/JSON evaluation outputs
│   ├── plots/                     # Generated charts and figures
│   └── human_eval/                # Human evaluation forms and results
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_chunking.py
│   └── test_evaluate.py
│
└── models/                        # Cached model weights (NOT committed to git)
```

## Dependencies

### requirements.txt

```
# ASR
openai-whisper>=20231117

# NLP / Transformers
transformers>=4.36.0
torch>=2.1.0
sentencepiece>=0.1.99
protobuf>=3.20.0

# Summarization baseline
sumy>=0.11.0
networkx>=3.2

# Text processing
nltk>=3.8.1
spacy>=3.7.0
regex>=2023.10.3

# Evaluation
rouge-score>=0.1.2
bert-score>=0.3.13

# Audio
librosa>=0.10.1
soundfile>=0.12.1
pydub>=0.25.1

# Data & config
pandas>=2.1.0
numpy>=1.24.0
pyyaml>=6.0.1
datasets>=2.16.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# Application
streamlit>=1.29.0

# Testing
pytest>=7.4.0

# Utilities
tqdm>=4.66.0
```

### Environment Setup Steps

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy English model
python -m spacy download en_core_web_sm

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# 5. Verify Whisper installation
python -c "import whisper; model = whisper.load_model('base'); print('Whisper OK')"

# 6. Verify transformers
python -c "from transformers import pipeline; print('Transformers OK')"
```

### Google Colab Setup (for GPU tasks)

```python
# Run at top of Colab notebook
!pip install openai-whisper transformers rouge-score bert-score sumy datasets tqdm

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Symlink project directory
import os
PROJECT_DIR = '/content/drive/MyDrive/MSML641_Project'
os.makedirs(PROJECT_DIR, exist_ok=True)
```

## .gitignore

```
# Data (too large for git)
data/raw/
data/transcripts/
data/processed/
models/

# Python
__pycache__/
*.pyc
*.pyo
venv/
.venv/
*.egg-info/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Keep sample data
!data/samples/
```

## Config File Format

### configs/config.yaml

```yaml
whisper:
  model_size: "base"          # tiny, base, small, medium, large
  language: "en"
  task: "transcribe"

preprocessing:
  remove_fillers: true
  filler_words:
    - "um"
    - "uh"
    - "like"
    - "you know"
    - "I mean"
    - "sort of"
    - "kind of"
    - "basically"
    - "actually"
    - "right"
  normalize_whitespace: true
  lowercase: false             # keep original case for summarization

summarization:
  extractive:
    method: "textrank"
    num_sentences: 5           # or ratio: 0.2
  abstractive:
    bart:
      model_name: "facebook/bart-large-cnn"
      max_input_tokens: 1024
      max_summary_length: 256
      min_summary_length: 56
      num_beams: 4
      length_penalty: 2.0
      no_repeat_ngram_size: 3
    t5:
      model_name: "t5-base"
      max_input_tokens: 512
      max_summary_length: 256
      min_summary_length: 56
      num_beams: 4
      length_penalty: 2.0
      prefix: "summarize: "

chunking:
  strategy: "sentence"         # sentence, token, overlap
  max_chunk_tokens: 800
  overlap_tokens: 100
  hierarchical: true           # two-pass summarization

evaluation:
  rouge_types:
    - "rouge1"
    - "rouge2"
    - "rougeL"
  use_bertscore: true
  bertscore_model: "roberta-large"

paths:
  raw_audio: "data/raw"
  transcripts: "data/transcripts"
  processed: "data/processed"
  reference_summaries: "data/reference_summaries"
  results: "results"
```
