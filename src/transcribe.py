"""
Whisper ASR transcription module.

Wraps OpenAI's Whisper model to convert audio files into timestamped
transcripts. Supports single-file and batch transcription with
configurable model sizes.

Typical usage::

    from src.transcribe import WhisperTranscriber

    transcriber = WhisperTranscriber(model_size="base")
    result = transcriber.transcribe("data/raw/meeting.wav")
    print(result["text"])
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

from src.utils import ensure_dir, setup_logger, write_json

logger = setup_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


class FFmpegNotFoundError(RuntimeError):
    """
    Raised when the ``ffmpeg`` executable is not on ``PATH``.

    OpenAI Whisper shells out to ``ffmpeg`` to decode compressed formats
    (MP3, M4A, etc.); without it, transcription fails with a cryptic
    ``FileNotFoundError``.
    """


def _ensure_ffmpeg_on_path() -> None:
    """
    If ``ffmpeg`` is missing from PATH, prepend common install locations (macOS).

    IDEs and GUI-launched terminals often omit Homebrew's ``bin`` directory, so
    ``brew install ffmpeg`` succeeds but Whisper still cannot spawn ``ffmpeg``.
    """
    if shutil.which("ffmpeg"):
        return
    import platform
    if platform.system() != "Darwin":
        return
    for candidate in (
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            bin_dir = str(candidate.parent)
            path = os.environ.get("PATH", "")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + path
            logger.info("Prepended %s to PATH so ffmpeg can be found", bin_dir)
            return


def _require_ffmpeg() -> None:
    """
    Ensure ``ffmpeg`` is available before Whisper tries to load audio.

    Raises:
        FFmpegNotFoundError: If ``ffmpeg`` is not installed or not on PATH.
    """
    _ensure_ffmpeg_on_path()
    if shutil.which("ffmpeg"):
        return
    raise FFmpegNotFoundError(
        "The `ffmpeg` program was not found on your PATH. Whisper uses ffmpeg to "
        "decode audio files (especially MP3/M4A).\n\n"
        "**Install ffmpeg, then restart your terminal and try again:**\n\n"
        "- **macOS (Homebrew):** `brew install ffmpeg`\n"
        "- **Ubuntu / Debian:** `sudo apt update && sudo apt install -y ffmpeg`\n"
        "- **Windows:** Install from https://ffmpeg.org/download.html and add "
        "`ffmpeg.exe` to your PATH.\n\n"
        "Verify: `ffmpeg -version`"
    )


class WhisperTranscriber:
    """
    Wrapper around OpenAI Whisper for audio transcription.

    Loads the model once on initialization and reuses it for all
    subsequent transcriptions, avoiding repeated model loading overhead.

    Args:
        model_size: Whisper model variant. One of:
                    ``tiny``, ``base``, ``small``, ``medium``, ``large``.
                    Larger models are slower but more accurate.
        device:     ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
        language:   ISO language code (e.g. ``"en"``). ``None`` for auto-detect.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = "en",
    ):
        import whisper

        self.model_size = model_size
        self.language = language

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading Whisper model '%s' on %s", model_size, device)
        self.model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file (.wav, .mp3, .m4a, etc.).
            language:   Override the default language for this call.

        Returns:
            Dictionary with keys:

            - ``audio_file`` (str): Basename of the input file.
            - ``text`` (str): Full transcript as a single string.
            - ``segments`` (list[dict]): Timestamped segments, each with
              ``start``, ``end``, and ``text``.
            - ``language`` (str): Detected or specified language code.
            - ``duration_seconds`` (float): Audio duration.
            - ``processing_time_seconds`` (float): Wall-clock inference time.
            - ``model_size`` (str): Whisper model variant used.
            - ``word_count`` (int): Number of words in the full transcript.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            ValueError:        If the file extension is not supported.
            FFmpegNotFoundError: If ``ffmpeg`` is not installed (see :func:`_require_ffmpeg`).
        """
        import whisper

        audio_path = Path(audio_path)
        self._validate_audio_path(audio_path)
        _require_ffmpeg()

        lang = language or self.language
        logger.info("Transcribing: %s (lang=%s)", audio_path.name, lang)

        start_time = time.perf_counter()

        result = self.model.transcribe(
            str(audio_path),
            language=lang,
            task="transcribe",
            verbose=False,
        )

        processing_time = time.perf_counter() - start_time

        audio_array = whisper.load_audio(str(audio_path))
        duration = len(audio_array) / whisper.audio.SAMPLE_RATE

        transcript_text = result["text"].strip()

        output = {
            "audio_file": audio_path.name,
            "text": transcript_text,
            "segments": [
                {
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
            "language": result.get("language", lang),
            "duration_seconds": round(duration, 2),
            "processing_time_seconds": round(processing_time, 2),
            "model_size": self.model_size,
            "word_count": len(transcript_text.split()),
        }

        logger.info(
            "Transcribed %s: %d words in %.1fs (audio=%.1fs)",
            audio_path.name,
            output["word_count"],
            processing_time,
            duration,
        )

        return output

    def transcribe_and_save(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        language: Optional[str] = None,
    ) -> Path:
        """
        Transcribe an audio file and save the result as JSON.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save the output JSON.
            language:   Override language for this call.

        Returns:
            Path to the saved JSON file.
        """
        result = self.transcribe(audio_path, language=language)
        output_dir = ensure_dir(output_dir)
        output_path = output_dir / f"{Path(audio_path).stem}.json"
        write_json(result, output_path)
        logger.info("Saved transcript to %s", output_path)
        return output_path

    def batch_transcribe(
        self,
        audio_dir: Union[str, Path],
        output_dir: Union[str, Path],
        language: Optional[str] = None,
    ) -> list[Path]:
        """
        Transcribe every supported audio file in a directory.

        Args:
            audio_dir:  Directory containing audio files.
            output_dir: Directory to save transcript JSONs.
            language:   Override language for all files.

        Returns:
            List of Paths to saved JSON transcript files.
        """
        from tqdm import tqdm

        audio_dir = Path(audio_dir)
        if not audio_dir.is_dir():
            raise NotADirectoryError(f"Audio directory not found: {audio_dir}")

        audio_files = sorted(
            f for f in audio_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )

        if not audio_files:
            logger.warning("No audio files found in %s", audio_dir)
            return []

        logger.info("Found %d audio files in %s", len(audio_files), audio_dir)
        output_paths = []

        for audio_file in tqdm(audio_files, desc="Transcribing"):
            try:
                path = self.transcribe_and_save(audio_file, output_dir, language)
                output_paths.append(path)
            except Exception as e:
                logger.error("Failed to transcribe %s: %s", audio_file.name, e)

        logger.info("Batch complete: %d/%d succeeded", len(output_paths), len(audio_files))
        return output_paths

    @staticmethod
    def transcribe_bytes(
        audio_bytes: bytes,
        model_size: str = "base",
        suffix: str = ".wav",
    ) -> dict:
        """
        Transcribe audio from raw bytes (useful for Streamlit file uploads).

        Writes bytes to a temp file, transcribes, then cleans up.

        Args:
            audio_bytes: Raw audio file content.
            model_size:  Whisper model size to use.
            suffix:      File extension hint for the temp file.

        Returns:
            Transcript dict (same format as :meth:`transcribe`).
        """
        _require_ffmpeg()
        transcriber = WhisperTranscriber(model_size=model_size)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return transcriber.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _validate_audio_path(audio_path: Path) -> None:
        """Raise clear errors for missing files or unsupported formats."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{audio_path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("input", help="Audio file or directory of audio files")
    parser.add_argument("-o", "--output", default="data/transcripts", help="Output directory")
    parser.add_argument("-m", "--model", default="base", help="Whisper model size")
    parser.add_argument("-l", "--language", default="en", help="Language code")
    args = parser.parse_args()

    transcriber = WhisperTranscriber(model_size=args.model, language=args.language)
    input_path = Path(args.input)

    if input_path.is_dir():
        transcriber.batch_transcribe(input_path, args.output)
    else:
        transcriber.transcribe_and_save(input_path, args.output)
