"""
Shared utilities for the Speech-to-Text Insights pipeline.

Provides config loading, logging setup, reproducibility (seed setting),
file I/O helpers, and timing context managers used across all modules.
"""

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project root — resolved once, reused everywhere
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """
    Load the YAML configuration file.

    If no path is given, loads the default at ``configs/config.yaml``.

    Args:
        config_path: Optional path to a YAML config file.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError:    If the file contains invalid YAML.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def get_nested(config: dict, *keys, default: Any = None) -> Any:
    """
    Safely retrieve a nested value from a config dict.

    Example::

        model_name = get_nested(cfg, "summarization", "abstractive", "bart", "model_name")

    Args:
        config: The config dictionary.
        *keys:  Sequence of keys to traverse.
        default: Value returned if any key is missing.

    Returns:
        The value at the nested path, or *default*.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create a configured logger with console (and optional file) output.

    Args:
        name:     Logger name — typically ``__name__`` of the calling module.
        level:    Logging level (default: INFO).
        log_file: If provided, also writes log output to this file path.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Call this at the start of every script / notebook.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_json(path: Union[str, Path]) -> Any:
    """
    Read a JSON file and return its parsed contents.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """
    Write data to a JSON file, creating parent directories as needed.

    Args:
        data:   JSON-serializable object.
        path:   Destination file path.
        indent: Pretty-print indentation (default 2).

    Returns:
        The resolved Path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)

    return path


def read_text(path: Union[str, Path]) -> str:
    """Read a text file and return its contents as a string."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8")


def write_text(text: str, path: Union[str, Path]) -> Path:
    """Write a string to a text file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager that logs elapsed wall-clock time.

    Usage::

        with timer("Whisper transcription"):
            result = transcribe(audio_path)

    Args:
        label:  Descriptive label printed alongside the timing.
        logger: If provided, logs at INFO level; otherwise prints.
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    msg = f"{label} completed in {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it doesn't exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(relative_path: str, base: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to the project root (or a custom base).

    Args:
        relative_path: Path string relative to *base*.
        base:          Base directory (defaults to PROJECT_ROOT).

    Returns:
        Resolved absolute Path.
    """
    base = base or PROJECT_ROOT
    return (base / relative_path).resolve()
