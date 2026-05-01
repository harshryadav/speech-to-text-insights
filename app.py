"""
Speech-to-Text Insights — Streamlit Application

Interactive web interface for uploading audio, viewing transcripts,
generating summaries, and comparing model performance.

Run with (from activated venv): python -m streamlit run app.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Speech-to-Text Insights",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _verify_runtime_environment() -> None:
    """
    Ensure Streamlit is using the same Python that has project dependencies.

    A common mistake is running `streamlit run app.py` when the `streamlit`
    on PATH belongs to a global Python install (e.g. macOS Framework Python)
    while `openai-whisper` was installed only inside `venv/`. That yields
    ModuleNotFoundError for `whisper` with no helpful hint in the UI.
    """
    try:
        import whisper  # noqa: F401 — openai-whisper package
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parent
        venv_python = project_root / "venv" / "bin" / "python"
        if venv_python.is_file():
            copy_paste = f'cd "{project_root}"\n"{venv_python}" -m streamlit run app.py'
        else:
            copy_paste = (
                f'cd "{project_root}"\n'
                "source venv/bin/activate\n"
                "python -m streamlit run app.py"
            )
        st.error(
            "#### Missing package: `whisper` (OpenAI Whisper)\n\n"
            "Streamlit is using a Python that is **not** your project `venv`, so "
            "`openai-whisper` is not installed for this interpreter.\n\n"
            f"**Interpreter in use (wrong for this project):**  \n`{sys.executable}`\n\n"
            "---\n\n"
            "**Fix A — copy/paste (uses your venv explicitly):**\n"
            "```bash\n"
            f"{copy_paste}\n"
            "```\n\n"
            "**Fix B — launcher script:**\n"
            "```bash\n"
            f"cd {project_root}\n"
            "chmod +x run_streamlit.sh   # once\n"
            "./run_streamlit.sh\n"
            "```\n\n"
            "**Fix C — Cursor:** Command Palette → **Python: Select Interpreter** → choose "
            f"`{venv_python}` (or `FInal_Project/venv/bin/python`), then restart Streamlit."
        )
        st.stop()


_verify_runtime_environment()


# ---------------------------------------------------------------------------
# Cached model loaders (persist across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Whisper model...")
def load_whisper(model_size: str):
    """Load and cache the Whisper transcriber."""
    from src.transcribe import WhisperTranscriber
    return WhisperTranscriber(model_size=model_size)


def try_load_hf_summarizer(model_name: str):
    """
    Load BART/T5 from Hugging Face. Caches only successful loads in session_state
    so a temporary VPN/network failure can be retried on the next run without a
    full app restart.

    Returns None if the model cannot be downloaded or configured (common on
    corporate VPNs that block huggingface.co).
    """
    import logging

    from src.summarize_abstractive import AbstractiveSummarizer

    log = logging.getLogger(__name__)
    cache = st.session_state.setdefault("_hf_summarizers_ok", {})
    if model_name in cache:
        return cache[model_name]

    try:
        summarizer = AbstractiveSummarizer(model_name=model_name)
        cache[model_name] = summarizer
        return summarizer
    except Exception as exc:
        log.warning("Could not load Hugging Face model %s: %s", model_name, exc)
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render sidebar controls and return user selections."""
    st.sidebar.title("🎙️ Speech-to-Text Insights")
    st.sidebar.markdown("---")

    # Audio upload
    audio_file = st.sidebar.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Supported formats: WAV, MP3, M4A, FLAC, OGG",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Settings")

    # Whisper model selection
    whisper_size = st.sidebar.selectbox(
        "Whisper Model Size",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower.",
    )

    # Summarization methods
    methods = st.sidebar.multiselect(
        "Summarization Methods",
        options=["TextRank", "BART", "T5"],
        default=["TextRank", "BART", "T5"],
        help="Select which summarizers to run.",
    )

    # Summary length
    summary_style = st.sidebar.radio(
        "Summary Length",
        options=["Brief", "Standard", "Detailed"],
        index=1,
    )

    length_map = {
        "Brief": {"max_length": 80, "min_length": 30, "num_sentences": 3},
        "Standard": {"max_length": 256, "min_length": 56, "num_sentences": 5},
        "Detailed": {"max_length": 512, "min_length": 128, "num_sentences": 10},
    }

    st.sidebar.markdown("---")

    # Optional reference summary
    reference_file = st.sidebar.file_uploader(
        "Reference Summary (optional)",
        type=["txt"],
        help="Upload a reference summary to compute ROUGE scores.",
    )

    # BERTScore is opt-in because the first run downloads roberta-large
    # (~1.4 GB) and can take 10–60s on CPU per pipeline run. Once cached,
    # subsequent runs in the same Streamlit session reuse the loaded model.
    compute_bertscore = st.sidebar.checkbox(
        "Compute BERTScore",
        value=False,
        help=(
            "Adds semantic similarity scores (captures paraphrase that ROUGE misses). "
            "First run downloads roberta-large (~1.4 GB) and is slow on CPU; "
            "later runs reuse the cached model. Requires a reference summary."
        ),
        disabled=reference_file is None,
    )

    # Process button
    process_btn = st.sidebar.button(
        "▶ Process Audio",
        type="primary",
        use_container_width=True,
        disabled=audio_file is None,
    )

    return {
        "audio_file": audio_file,
        "whisper_size": whisper_size,
        "methods": [m.lower() for m in methods],
        "summary_params": length_map[summary_style],
        "reference_file": reference_file,
        "compute_bertscore": compute_bertscore,
        "process": process_btn,
    }


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def process_audio(settings: dict) -> dict:
    """Run the full pipeline on uploaded audio and return results."""
    from src.preprocess import preprocess_transcript
    from src.chunking import chunk_text
    from src.summarize_extractive import textrank_summarize
    from src.evaluate import compute_rouge
    from src.transcribe import FFmpegNotFoundError

    results = {"timings": {}}
    audio_file = settings["audio_file"]
    params = settings["summary_params"]

    # --- Transcription ---
    with st.spinner("Transcribing audio with Whisper..."):
        transcriber = load_whisper(settings["whisper_size"])

        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(audio_file.name)[1],
            delete=False,
        ) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            t0 = time.perf_counter()
            try:
                transcript = transcriber.transcribe(tmp_path)
            except FFmpegNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            results["timings"]["transcription"] = round(time.perf_counter() - t0, 2)
        finally:
            os.unlink(tmp_path)

    results["transcript"] = transcript

    # --- Preprocessing ---
    with st.spinner("Preprocessing transcript..."):
        t0 = time.perf_counter()
        preprocessed = preprocess_transcript(transcript["text"])
        results["timings"]["preprocessing"] = round(time.perf_counter() - t0, 2)

    results["preprocessing"] = preprocessed

    # --- Chunking ---
    chunks = chunk_text(
        preprocessed["cleaned"],
        sentences=preprocessed["sentences"],
        max_tokens=800,
    )
    results["chunks"] = chunks

    # --- Summarization ---
    summaries = {}
    methods = settings["methods"]
    hf_skipped: list[str] = []

    if "textrank" in methods:
        with st.spinner("Running TextRank..."):
            t0 = time.perf_counter()
            summaries["textrank"] = textrank_summarize(
                preprocessed["cleaned"],
                num_sentences=params["num_sentences"],
            )
            results["timings"]["textrank"] = round(time.perf_counter() - t0, 2)

    if "bart" in methods:
        with st.spinner("Loading BART and summarizing..."):
            summarizer = try_load_hf_summarizer("facebook/bart-large-cnn")
            if summarizer is None:
                hf_skipped.append("BART (`facebook/bart-large-cnn`)")
            else:
                t0 = time.perf_counter()
                summaries["bart"] = summarizer.summarize_long(
                    chunks,
                    hierarchical=True,
                    max_length=params["max_length"],
                    min_length=params["min_length"],
                )
                results["timings"]["bart"] = round(time.perf_counter() - t0, 2)

    if "t5" in methods:
        with st.spinner("Loading Flan-T5 and summarizing..."):
            summarizer = try_load_hf_summarizer("google/flan-t5-base")
            if summarizer is None:
                hf_skipped.append("T5 (`google/flan-t5-base`)")
            else:
                t0 = time.perf_counter()
                summaries["t5"] = summarizer.summarize_long(
                    chunks,
                    hierarchical=True,
                    max_length=params["max_length"],
                    min_length=params["min_length"],
                )
                results["timings"]["t5"] = round(time.perf_counter() - t0, 2)

    if hf_skipped:
        st.warning(
            "Could not load from Hugging Face (often blocked on a work VPN): "
            + ", ".join(hf_skipped)
            + ". **TextRank** still works offline. Fixes: try without VPN, use home Wi‑Fi, "
            "ask IT to allow **huggingface.co**, or copy the Hugging Face cache from a "
            "machine that can download (see README → Hugging Face / VPN)."
        )

    results["summaries"] = summaries

    # --- Evaluation (if reference provided) ---
    if settings["reference_file"]:
        reference = settings["reference_file"].read().decode("utf-8").strip()
        evaluation = {}
        for method, summary in summaries.items():
            try:
                evaluation[method] = compute_rouge(summary, reference)
            except ValueError:
                evaluation[method] = {"error": "Could not compute ROUGE"}

        # Optional BERTScore — opt-in because the embedding model is ~1.4 GB
        # and cold-start can take a while on CPU. Computed once for all
        # summaries to share the model load.
        if settings.get("compute_bertscore"):
            from src.evaluate import compute_bertscore_per_pair

            scorable = [
                (m, s) for m, s in summaries.items()
                if s and "error" not in evaluation.get(m, {})
            ]
            if scorable:
                with st.spinner(
                    "Computing BERTScore (first run downloads roberta-large)..."
                ):
                    t0 = time.perf_counter()
                    try:
                        per_pair = compute_bertscore_per_pair(
                            predictions=[s for _, s in scorable],
                            references=[reference] * len(scorable),
                        )
                        for (method, _), bs in zip(scorable, per_pair):
                            evaluation[method]["bertscore"] = bs
                        results["timings"]["bertscore"] = round(
                            time.perf_counter() - t0, 2
                        )
                    except Exception as exc:
                        st.warning(
                            f"BERTScore could not be computed: {exc}. "
                            "ROUGE results above are unaffected."
                        )

        results["evaluation"] = evaluation
        results["reference"] = reference
    else:
        results["evaluation"] = None

    return results


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def render_transcript_tab(results: dict):
    """Render the transcript comparison tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Transcript")
        st.text_area(
            "Raw",
            value=results["transcript"]["text"],
            height=400,
            label_visibility="collapsed",
        )

    with col2:
        st.subheader("Cleaned Transcript")
        st.text_area(
            "Cleaned",
            value=results["preprocessing"]["cleaned"],
            height=400,
            label_visibility="collapsed",
        )

    # Stats row
    stats = results["preprocessing"]["stats"]
    cols = st.columns(4)
    cols[0].metric("Original Words", stats["original_word_count"])
    cols[1].metric("Cleaned Words", stats["cleaned_word_count"])
    cols[2].metric("Fillers Removed", stats["fillers_removed"])
    cols[3].metric("Sentences", stats["sentence_count"])


def render_summaries_tab(results: dict):
    """Render the summaries comparison tab."""
    summaries = results["summaries"]

    if not summaries:
        st.info("No summaries generated. Select at least one method.")
        return

    cols = st.columns(len(summaries))
    for col, (method, summary) in zip(cols, summaries.items()):
        with col:
            label = {"textrank": "TextRank (Extractive)", "bart": "BART (Abstractive)", "t5": "Flan-T5 (Abstractive)"}
            st.subheader(label.get(method, method.upper()))
            st.write(summary)
            word_count = len(summary.split())
            st.caption(f"{word_count} words")


def render_evaluation_tab(results: dict):
    """Render the evaluation tab with ROUGE scores."""
    if results["evaluation"] is None:
        st.info(
            "Upload a reference summary in the sidebar to see evaluation scores."
        )
        return

    import pandas as pd

    eval_data = results["evaluation"]
    rows = []
    any_bertscore = False
    for method, scores in eval_data.items():
        if "error" in scores:
            continue
        row = {
            "Model": method.upper(),
            "ROUGE-1 (F1)": scores["rouge1"]["f1"],
            "ROUGE-2 (F1)": scores["rouge2"]["f1"],
            "ROUGE-L (F1)": scores["rougeL"]["f1"],
        }
        if "bertscore" in scores:
            row["BERTScore (F1)"] = scores["bertscore"]["f1"]
            any_bertscore = True
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart includes BERTScore as a fourth metric group when available.
        import plotly.express as px

        df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(
            df_melted,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title=(
                "ROUGE + BERTScore Comparison"
                if any_bertscore
                else "ROUGE Score Comparison"
            ),
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        if any_bertscore:
            st.caption(
                "BERTScore measures semantic similarity via contextual embeddings "
                "(roberta-large). It tends to credit paraphrase that ROUGE misses, "
                "so abstractive models (BART/T5) often score relatively higher here "
                "than on ROUGE."
            )
        elif results["evaluation"]:
            st.caption(
                "Tip: enable **Compute BERTScore** in the sidebar for a semantic "
                "similarity score that complements ROUGE's n-gram overlap."
            )


def render_stats_tab(results: dict):
    """Render the statistics tab."""
    timings = results["timings"]

    st.subheader("Processing Times")
    cols = st.columns(len(timings))
    for col, (stage, seconds) in zip(cols, timings.items()):
        col.metric(stage.capitalize(), f"{seconds:.1f}s")

    st.subheader("Document Statistics")
    transcript = results["transcript"]
    cols = st.columns(3)
    cols[0].metric("Audio Duration", f"{transcript['duration_seconds']:.1f}s")
    cols[1].metric("Transcript Words", transcript["word_count"])
    cols[2].metric("Chunks Created", len(results["chunks"]))

    # Compression ratios
    if results["summaries"]:
        st.subheader("Compression Ratios")
        original_words = transcript["word_count"]
        cols = st.columns(len(results["summaries"]))
        for col, (method, summary) in zip(cols, results["summaries"].items()):
            summary_words = len(summary.split())
            ratio = round(summary_words / max(original_words, 1) * 100, 1)
            col.metric(
                method.upper(),
                f"{ratio}%",
                f"{summary_words} words",
            )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit application entry point."""
    settings = render_sidebar()

    # Header
    st.title("🎙️ Speech-to-Text Insights")
    st.markdown(
        "Upload an audio file to get an automatic transcript and AI-generated summaries."
    )

    if not settings["process"]:
        st.markdown("---")
        st.markdown(
            """
            ### How to use
            1. **Upload** an audio file using the sidebar
            2. **Choose** your Whisper model size and summarization methods
            3. **Click** "Process Audio" to run the pipeline
            4. **Explore** the tabs to see transcripts, summaries, and evaluation
            """
        )
        return

    # Run pipeline
    results = process_audio(settings)

    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Transcript", "📋 Summaries", "📊 Evaluation", "⚙️ Statistics"]
    )

    with tab1:
        render_transcript_tab(results)

    with tab2:
        render_summaries_tab(results)

    with tab3:
        render_evaluation_tab(results)

    with tab4:
        render_stats_tab(results)

    # Download button
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        "📥 Download Results (JSON)",
        data=json.dumps(
            {
                "transcript": results["transcript"]["text"],
                "cleaned": results["preprocessing"]["cleaned"],
                "summaries": results["summaries"],
                "stats": results["preprocessing"]["stats"],
                "timings": results["timings"],
            },
            indent=2,
        ),
        file_name="speech_insights_results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
