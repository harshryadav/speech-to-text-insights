# 08 — Risk Mitigation & Troubleshooting

---

## 1. Risk Register

```
┌────────────────────────────────────────────────────────────────────────┐
│                         RISK MATRIX                                    │
│                                                                        │
│  Impact ▲                                                              │
│         │                                                              │
│   HIGH  │  [R3] GPU unavailable   [R1] Token limits    [R6] Low ROUGE │
│         │  for BART/T5            blow up pipeline      despite effort │
│         │                                                              │
│   MED   │  [R5] AMI download      [R2] ASR errors      [R7] Colab    │
│         │  issues                  degrade summaries    timeouts      │
│         │                                                              │
│   LOW   │  [R8] Team member       [R4] Noisy audio     [R9] Streamlit│
│         │  unavailable            poor transcription    deployment    │
│         │                                                              │
│         └──────────────────────────────────────────────────────────►   │
│              LOW                    MED                    HIGH        │
│                              Likelihood                                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Risk Analysis & Mitigation

### R1: Long Transcripts Exceed Token Limits

**Risk**: BART (1024 tokens) and T5 (512 tokens) cannot process full meeting transcripts (often 2000-5000 tokens). Naive truncation loses critical information.

**Likelihood**: HIGH (virtually guaranteed for meetings > 5 min)

**Mitigation**:
- Sentence-aware chunking is already planned (see `src/chunking.py`)
- Hierarchical summarization (two-pass) handles chunk count overflow
- Test edge cases: very short audio (< 30s), very long audio (> 1 hour)

**Detection**: Log chunk counts. If any transcript produces > 10 chunks, investigate.

**Fallback**: If chunking produces poor results, try a larger context window model like `LED` (Longformer Encoder-Decoder) which handles 16K tokens.

```python
# Fallback: Longformer Encoder-Decoder
# pip install transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "allenai/led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Accepts up to 16,384 tokens — no chunking needed for most transcripts
```

---

### R2: ASR Errors Propagate to Summaries

**Risk**: Whisper misrecognizes words (especially names, jargon, accented speech). These errors flow into summaries, potentially changing meaning.

**Likelihood**: MEDIUM

**Examples**:
- "We should buy more servers" → "We should buy more service" (phonetic confusion)
- Proper nouns mangled: "Let's ask Dr. Patel" → "Let's ask Dr. Patel" (usually fine) vs "Let's ask Doctor Patrol"
- Technical terms: "Kubernetes" → "Cooper Nettie's"

**Mitigation**:
1. Use larger Whisper models (medium/large) for better accuracy
2. Preprocessing step: spell-check against a domain dictionary
3. Run ablation A3 (Whisper model size) to quantify the impact
4. In the report, honestly discuss ASR error propagation as a limitation

**Detection**: Compare Whisper output against AMI's manual transcripts. Compute Word Error Rate (WER).

---

### R3: GPU Unavailable for Model Inference

**Risk**: BART-large and Whisper-medium require GPU for reasonable speed. Google Colab free tier limits GPU access.

**Likelihood**: MEDIUM

**Mitigation**:
- **Primary**: Use Google Colab free tier (T4 GPU, 12GB VRAM). Schedule heavy jobs during off-peak hours.
- **Backup 1**: Use smaller models (`whisper-base`, `t5-small`, `bart-base` instead of `bart-large-cnn`)
- **Backup 2**: Run on CPU — slower but works. BART-large on CPU: ~30-60s per summary (tolerable for 10-20 test samples)
- **Backup 3**: Colab Pro ($10/month) if free tier is insufficient
- **Backup 4**: University HPC (ask instructor for access)

**CPU Feasibility**:
| Model | GPU (T4) per sample | CPU per sample | Feasible? |
|-------|-------------------|----------------|-----------|
| Whisper-base | ~5s | ~30s | Yes |
| Whisper-medium | ~15s | ~120s | Marginal |
| BART-large-cnn | ~3s | ~30s | Yes |
| T5-base | ~2s | ~20s | Yes |

For 20 test samples: CPU total ~30 minutes for all models. Acceptable.

---

### R4: Noisy Real-World Audio

**Risk**: Custom recordings have background noise, cross-talk, poor microphone quality.

**Likelihood**: HIGH (intentional — we're testing robustness)

**Mitigation**:
- Whisper is designed for robustness to noise (trained on noisy web data)
- Document noise conditions for each custom recording
- Compare clean vs noisy audio in the report
- If severely noisy: preprocess with `noisereduce` library

```python
# Audio noise reduction (optional preprocessing)
import noisereduce as nr
import librosa

audio, sr = librosa.load("noisy_audio.wav", sr=16000)
reduced = nr.reduce_noise(y=audio, sr=sr)
# Save and pass to Whisper
```

---

### R5: AMI Corpus Download Issues

**Risk**: The AMI corpus is large (~100 GB for full audio). Downloads may be slow or fail. HuggingFace hosting may have issues.

**Likelihood**: MEDIUM

**Mitigation**:
1. **Start download early** (Day 1 of Phase 2)
2. If full download fails, use only the test split (~10 meetings)
3. Alternative: Use HuggingFace `datasets` streaming mode:

```python
# Stream without downloading entire dataset
from datasets import load_dataset
ds = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
for sample in ds:
    # Process one at a time
    break
```

4. **Worst case fallback**: Use TED-LIUM or even just custom recordings + manually written reference summaries

---

### R6: ROUGE Scores Are Lower Than Expected

**Risk**: Meeting transcripts are noisy and informal. Reference summaries may be abstractive and use very different wording. Result: low ROUGE even for good summaries.

**Likelihood**: MEDIUM

**Mitigation**:
1. **Supplement with BERTScore** — captures semantic similarity even when wording differs
2. **Add human evaluation** — demonstrates quality beyond what ROUGE captures
3. **Frame results carefully**: In the report, explain that ROUGE is known to correlate weakly with human judgment for abstractive summarization (cite: Fabbri et al., 2021, "SummEval")
4. **Focus on relative improvement**: Even if absolute ROUGE is low, showing BART >> TextRank is the key finding

---

### R7: Google Colab Session Timeouts

**Risk**: Colab disconnects after 90 minutes of inactivity (free tier) or 12 hours of continuous use. Long batch jobs may be interrupted.

**Likelihood**: MEDIUM

**Mitigation**:
1. **Save intermediate results** after each sample (don't keep everything in memory)
2. **Use checkpointing**: write results to Google Drive after each batch
3. **Colab keep-alive**: add a JavaScript snippet to prevent idle disconnection

```python
# Save results incrementally
import json

results_file = "results/incremental_results.json"

for i, sample in enumerate(test_samples):
    result = process_sample(sample)

    # Load existing results, append, save
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(result)

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved {i+1}/{len(test_samples)}")
```

---

### R8: Team Member Unavailable

**Risk**: Illness, other deadlines, personal issues reduce team capacity.

**Likelihood**: LOW

**Mitigation**:
- Both members should understand the full pipeline (no knowledge silos)
- All code is in shared repo with clear documentation
- Tasks are assigned with overlap — either person can pick up the other's work
- Communicate early if falling behind

---

### R9: Streamlit Deployment Issues

**Risk**: Streamlit app works locally but crashes on presentation day (different Python version, missing model weights, audio codec issues).

**Likelihood**: LOW

**Mitigation**:
1. Test on the presentation machine at least 2 days before
2. Pre-cache model weights (don't download during demo)
3. Have pre-computed results as fallback (show screenshots if app crashes)
4. Use `streamlit run app.py` with pinned dependency versions

---

## 3. Common Error Solutions

### "CUDA out of memory"
```python
# Reduce batch size or use CPU
import torch
torch.cuda.empty_cache()

# Use half-precision (FP16) for inference
model = model.half()

# Or force CPU
model = model.to("cpu")
```

### "Tokenizer produces too many tokens"
```python
# Ensure truncation is on
inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
```

### "Whisper takes too long on CPU"
```python
# Use whisper-tiny or whisper-base (not medium/large) on CPU
model = whisper.load_model("base")

# Or transcribe shorter segments
result = model.transcribe(audio_path, verbose=True)  # see progress
```

### "ROUGE scores are all zero"
```python
# Check that prediction and reference are not empty strings
assert len(prediction.strip()) > 0, "Empty prediction"
assert len(reference.strip()) > 0, "Empty reference"

# Check that you're not swapping prediction and reference
scores = scorer.score(reference, prediction)  # reference FIRST
```

### "sumy TextRank produces empty summary"
```python
# Ensure enough sentences in input
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

parser = PlaintextParser.from_string(text, Tokenizer("english"))
if len(parser.document.sentences) < num_sentences:
    num_sentences = len(parser.document.sentences)
```

### "Audio file format not supported by Whisper"
```python
# Convert to WAV first
from pydub import AudioSegment

audio = AudioSegment.from_file("input.m4a")
audio = audio.set_frame_rate(16000).set_channels(1)
audio.export("input_converted.wav", format="wav")
```
