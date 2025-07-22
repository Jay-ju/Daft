# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, cast

import ffmpeg
import numpy as np  # noqa: TID253
import soundfile as sf
import torch
from torchcodec.decoders import AudioDecoder

from daft.las.functions.utils.common_utils import load_file

logger = logging.getLogger(__name__)


def get_duration(local_file_path: str) -> float:
    probe = ffmpeg.probe(local_file_path)
    return float(probe["format"]["duration"])


def decode_audio(
    source: str | bytes,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> AudioDecoder:
    """Decode audio from local/remote path or raw bytes into an AudioDecoder.

    Supported sources:
      - Raw audio bytes
      - Local file paths
      - HTTP/TOS/S3 URIs

    Optionally resamples the audio and/or adjusts number of channels.

    Args:
        source: Input audio source, either bytes or a path-like string.
        sample_rate: Optional target sampling rate for resampling (e.g., 16000).
        num_channels: Optional target number of channels (e.g., 1 for mono, 2 for stereo).

    Returns:
        Decoded audio as an AudioDecoder object.
    """
    raw_bytes = cast("bytes", load_file(source))
    return AudioDecoder(BytesIO(raw_bytes), sample_rate=sample_rate, num_channels=num_channels)


def encode_audio(
    audio: AudioDecoder | dict[str, Any], file_format: str = "WAV", as_base64: bool = False
) -> bytes | str:
    """Encodes audio to WAV format from either an AudioDecoder or a dict.

    Args:
        audio (AudioDecoder or dict): The source audio.
            - If AudioDecoder: uses `get_all_samples()`
            - If dict: expects keys "samples" and "sample_rate"
        as_base64 (bool, optional): Whether to return base64-encoded string.

    Returns:
        bytes | str: WAV-encoded audio as bytes or base64 string.
    """
    # Step 1: Extract waveform and sample_rate
    if isinstance(audio, AudioDecoder):
        samples = audio.get_all_samples()
        array = samples.data.cpu().numpy()
        sample_rate = samples.sample_rate
    elif isinstance(audio, dict):
        array = audio["samples"]
        sample_rate = audio["sample_rate"]
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()
        elif isinstance(array, list):
            array = np.array(array, dtype=np.float32)
        if array.ndim == 1:
            array = array[np.newaxis, :]  # mono: (1, N)
    else:
        raise TypeError("Expected audio to be AudioDecoder or dict")

    # Step 2: Encode to WAV in-memory
    with BytesIO() as buffer:
        sf.write(buffer, array.T, sample_rate, format=file_format)
        wav_bytes = buffer.getvalue()

    # Step 3: Return as base64 or raw bytes
    if as_base64:
        return base64.b64encode(wav_bytes).decode("utf-8")
    return wav_bytes
