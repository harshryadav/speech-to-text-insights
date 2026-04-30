# 07 — Enhancements & Bonus Ideas

**Constraint**: ~3 weeks of remaining work. Each enhancement below is labeled with estimated effort and impact. Prioritize high-impact, low-effort items.

---

## Priority Legend

| Label | Effort | Time Estimate |
|-------|--------|---------------|
| LOW effort | < 2 hours | Quick win |
| MEDIUM effort | 2-6 hours | Half-day task |
| HIGH effort | 6-16 hours | 1-2 day task |

---

## Tier 1: Strongly Recommended (High Impact, Achievable)

### E1. Speaker Diarization
**Effort**: MEDIUM | **Impact**: HIGH | **Bonus Points**: Yes (innovation)

Add "who said what" to the transcript using Whisper's word-level timestamps + a diarization model.

```
Before: "We should increase the budget. I disagree, the budget is fine."
After:  "[Speaker A] We should increase the budget.
         [Speaker B] I disagree, the budget is fine."
```

**Implementation**:
```python
# Use pyannote.audio for speaker diarization
# pip install pyannote.audio

from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"  # requires HuggingFace agreement
)

diarization = diarization_pipeline("audio.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{speaker}] {turn.start:.1f}s - {turn.end:.1f}s")
```

**Why it helps**: Adds structure to meeting transcripts. Summaries can mention "Speaker A proposed X, Speaker B disagreed." This is a strong differentiator from basic pipelines.

**Fallback**: If `pyannote` is too complex, use Whisper's built-in `--initial_prompt` with speaker hints, or simply segment by pause duration.

---

### E2. Flan-T5 (Instruction-Tuned Model)
**Effort**: LOW | **Impact**: MEDIUM | **Bonus Points**: Yes (additional model comparison)

Replace or supplement `t5-base` with `google/flan-t5-base` — an instruction-tuned version that tends to produce better zero-shot summaries.

```python
# Drop-in replacement — same API
summarizer = AbstractiveSummarizer("google/flan-t5-base")
```

**Why it helps**: Flan-T5 was fine-tuned on 1.8K tasks including summarization. It often outperforms vanilla T5 without any additional effort. Adding this as a fourth model strengthens the comparison section.

---

### E3. Key Action Items / Topic Extraction
**Effort**: MEDIUM | **Impact**: HIGH | **Bonus Points**: Yes (deployment/innovation)

Beyond just summarization, extract structured insights:
- **Action items**: "John will send the report by Friday"
- **Key decisions**: "The team decided to proceed with option B"
- **Topics discussed**: ["budget", "marketing strategy", "hiring plan"]

**Implementation approach**:
```python
# Option A: Use a zero-shot classifier for topic extraction
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

topics = classifier(
    transcript_text,
    candidate_labels=["budget", "strategy", "hiring", "timeline", "technical"],
    multi_label=True
)

# Option B: Use regex + NLP patterns for action items
import re
action_patterns = [
    r"(?:will|shall|going to|need to|should)\s+(.+?)(?:\.|$)",
    r"(?:action item|todo|task)[\s:]+(.+?)(?:\.|$)",
]
```

**Why it helps**: Transforms the app from "just another summarizer" into a meeting insights tool. Directly addresses the "Speech-to-Text **Insights**" project name. Strong demo value.

---

### E4. Summary Quality Comparison Dashboard
**Effort**: MEDIUM | **Impact**: HIGH | **Bonus Points**: Yes (visualization)

Add a dedicated comparison view to the Streamlit app that shows:
- Side-by-side summaries with highlighted differences
- ROUGE score radar chart
- Word cloud of each summary
- Compression ratio comparison

```python
# Word cloud for visual comparison
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text: str, title: str):
    wc = WordCloud(width=400, height=200, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title)
    ax.axis("off")
    return fig
```

---

## Tier 2: Nice to Have (Medium Impact, Moderate Effort)

### E5. Sliding Window with Importance Weighting
**Effort**: MEDIUM | **Impact**: MEDIUM

Instead of uniform chunking, weight chunks by importance. Give more "summary budget" to chunks with higher information density.

```python
def importance_weighted_summarize(chunks, model, max_total_length=300):
    """
    1. Compute importance score for each chunk (e.g., keyword density, TF-IDF)
    2. Allocate max_length proportionally to importance
    3. Summarize each chunk with its allocated length
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    importance_scores = tfidf_matrix.sum(axis=1).A1
    importance_scores = importance_scores / importance_scores.sum()

    summaries = []
    for chunk, weight in zip(chunks, importance_scores):
        chunk_max_length = max(30, int(max_total_length * weight))
        summary = model.summarize(chunk, max_length=chunk_max_length)
        summaries.append(summary)

    return " ".join(summaries)
```

---

### E6. Multi-Document Summarization
**Effort**: MEDIUM | **Impact**: MEDIUM

Allow summarizing multiple meeting recordings into a single executive summary (e.g., weekly meeting digest).

```
Meeting 1 transcript → Summary 1 ─┐
Meeting 2 transcript → Summary 2 ─┼──► Combined Summary
Meeting 3 transcript → Summary 3 ─┘    "This week: ..."
```

**Implementation**: Concatenate individual summaries, then run a final summarization pass with a prompt like "Summarize the key themes from these meeting notes."

---

### E7. Configurable Summary Length
**Effort**: LOW | **Impact**: MEDIUM

Let users choose summary style:
- **Brief**: 2-3 sentences (executive summary)
- **Standard**: 5-7 sentences (default)
- **Detailed**: 10-15 sentences (comprehensive)

```python
LENGTH_PRESETS = {
    "brief":    {"max_length": 80,  "min_length": 30,  "num_sentences": 3},
    "standard": {"max_length": 256, "min_length": 56,  "num_sentences": 7},
    "detailed": {"max_length": 512, "min_length": 128, "num_sentences": 15},
}
```

Add a radio button in Streamlit sidebar.

---

### E8. Export Functionality
**Effort**: LOW | **Impact**: LOW-MEDIUM

Add download buttons for:
- Full transcript (`.txt`)
- Summary (`.txt`)
- Evaluation results (`.csv`)
- Full report (`.json` with all metadata)
- Timestamped transcript (`.srt` subtitle format)

```python
# SRT format for subtitle export
def transcript_to_srt(segments: list[dict]) -> str:
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
    return "\n".join(srt_lines)
```

---

## Tier 3: Stretch Goals (High Effort, Consider Only if Ahead of Schedule)

### E9. Fine-Tune BART on AMI Summaries
**Effort**: HIGH | **Impact**: HIGH

Fine-tune `facebook/bart-large-cnn` on the AMI corpus's transcript-summary pairs. This would likely produce the best ROUGE scores, but requires GPU hours and careful training.

**Only attempt if**: All other deliverables are complete by May 5 and GPU compute is available (Colab Pro or university HPC).

---

### E10. Real-Time Streaming Transcription
**Effort**: HIGH | **Impact**: MEDIUM (demo wow-factor)

Use Whisper in streaming mode to transcribe live microphone input and generate rolling summaries.

**Only attempt if**: Everything else is done and you want a dramatic demo.

---

## Recommendation: Pick 2-3 from Tier 1

Given the 3-week constraint:

| Enhancement | When to implement | Estimated effort |
|-------------|-------------------|------------------|
| **E2: Flan-T5** | Phase 3 (Apr 8-22) | 1-2 hours |
| **E7: Configurable length** | Phase 4 (Apr 22-May 1) | 1 hour |
| **E3: Key action items** | Phase 4 (Apr 25-30) | 4-6 hours |
| **E4: Comparison dashboard** | Phase 4 (Apr 28-May 2) | 4-6 hours |

Total additional effort: ~12-15 hours across the entire remaining timeline.

These four enhancements together would earn bonus points for:
- "Deployment of working system" (Streamlit app with rich features)
- "Novel dataset creation" or "Publication-quality analysis" (if the comparison is thorough)
- Innovation in the rubric (action item extraction goes beyond basic summarization)
