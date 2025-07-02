# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import io
from typing import Any

import librosa
import numpy as np  # noqa: TID253
import pyarrow as pa  # noqa: TID253
import soundfile as sf

from daft.las.functions.utils.common_utils import run_on_local_path


def load_audios(sources: Any, sr: int | None, mono: bool = False) -> list[tuple[np.ndarray, int]]:
    """Loads multiple audio sources.

    Args:
        sources (list): List of audio sources (see `load_audio` for supported types).
        sr (int, optional): Target sample rate. If None, preserves original.
        mono (bool, optional): Whether to convert to mono.

    Returns:
        list[tuple[np.ndarray, int]]: List of (audio, sample_rate) tuples.
    """
    return [load_audio(source, sr=sr, mono=mono) for source in sources]


def load_audio(
    source: Any,
    sr: int | None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Load audio from file path, URL, base64 string, TOS path, or (ndarray, sr) tuple.

    Args:
        source (str | tuple[np.ndarray, int] | dict | pa.lib.StructScalar):
            Path/URL/base64/TOS to audio file, or (array, sample_rate) tuple,
            or dict/StructScalar with "data" and "sample_rate".
        sr (int, optional): Target sample rate. If None, preserves original.
        mono (bool, optional): Whether to convert to mono.

    Returns:
        tuple[np.ndarray, int]: Audio time series and sample rate.

    Raises:
        TypeError: If input type is not supported.

    Example:
        - File path: "audio.wav"
        - Base64: "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0Y..."
        - Tuple: (np.ndarray, 16000)
        - Dict: {"data": np.ndarray, "sample_rate": 16000}
    """

    def is_base64(s: str) -> bool:
        try:
            base64.b64decode(s, validate=True)
        except Exception:
            return False
        else:
            return True

    def extract_base64_data(s: str) -> str:
        # Handles base64 strings like "data:audio/wav;base64,..." or just base64 content.
        # Example: "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0Y..."
        if s.startswith("data:"):
            return s.split(",")[1]
        return s

    # Enforce (ndarray, sr) only
    if isinstance(source, tuple) and len(source) == 2:
        y, orig_sr = source
        if not isinstance(y, np.ndarray) or not isinstance(orig_sr, int):
            raise TypeError("If passing a tuple, it must be (np.ndarray, int).")
        if sr and sr != orig_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        if mono and y.ndim > 1:
            y = np.mean(y, axis=0)
        return y, sr or orig_sr

    if (
        (isinstance(source, dict) or isinstance(source, pa.lib.StructScalar))
        and "data" in source
        and "sample_rate" in source
    ):
        if isinstance(source, pa.lib.StructScalar):
            source = source.as_py()
        y = source["data"]
        orig_sr = source["sample_rate"]

        if isinstance(y, list):
            y = np.array(y)
        if not isinstance(y, np.ndarray) or not isinstance(orig_sr, int):
            raise TypeError("If passing a tuple, it must be (np.ndarray, int).")
        if sr and sr != orig_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        if mono and y.ndim > 1:
            y = np.mean(y, axis=0)
        return y, sr or orig_sr

    if isinstance(source, np.ndarray):
        raise TypeError("Raw np.ndarray must be passed as a (ndarray, sample_rate) tuple.")

    if isinstance(source, str):
        if source.strip().startswith("data:audio") or is_base64(source):
            b64_data = extract_base64_data(source)
            audio_bytes = base64.b64decode(b64_data)
            with io.BytesIO(audio_bytes) as buf:
                data, rate = sf.read(buf)
                y = librosa.resample(data.T if data.ndim > 1 else data, orig_sr=rate, target_sr=sr) if sr else data
                if mono and y.ndim > 1:
                    y = np.mean(y, axis=0)
                return y, sr or rate

        def load(local_path: str) -> tuple[np.ndarray, int]:
            data, rate = librosa.load(local_path, sr=sr, mono=mono)
            y = librosa.resample(data.T if data.ndim > 1 else data, orig_sr=rate, target_sr=sr) if sr else data
            if mono and y.ndim > 1:
                y = np.mean(y, axis=0)
            return y, sr or rate

        return run_on_local_path(source, lambda path: load(path))

    raise TypeError("Unsupported source type. Must be str or (ndarray, int).")


def get_duration(local_file_path: str) -> float:
    """Get the duration of an audio file.

    Args:
        local_file_path (str): Path to the local audio file.

    Returns:
        float: Duration in seconds.
    """
    return librosa.get_duration(path=local_file_path)
