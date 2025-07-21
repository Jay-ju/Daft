# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

import numpy as np  # noqa: TID253
import soundfile as sf
import torch
from torchcodec.decoders import AudioDecoder

from daft.las.functions.utils.common_utils import run_on_local_path

logger = logging.getLogger(__name__)


def decode_audio(
    source: bytes | str,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> AudioDecoder:
    """Decode audio from various sources and formats.

    This function supports decoding audio from both pure audio and video files,
    including formats like `.mp3`, `.wav`, `.flac`, `.mp4`, and `.mov`. It accepts
    multiple types of input sources, such as:

      - Raw audio bytes (e.g., from memory or blob storage).
      - Local file paths.
      - HTTP/HTTPS URLs.
      - TOS/S3 URIs.

    It automatically handles downloading and decoding based on the file type.

    Args:
        source (bytes | str): Audio input to decode. Can be:
            - Raw bytes of audio content.
            - A string path pointing to a local file or remote URI.
        sample_rate (int | None, optional): Target sampling rate in Hz. If None,
            the original sampling rate is preserved. Defaults to None.
        num_channels (int | None, optional): Target number of audio channels.
            Can be 1 (mono), 2 (stereo), or None to preserve the original. Defaults to None.

    Returns:
        AudioDecoder: An object representing the decoded audio data, including waveform
        and metadata.

    """

    def decoder(source: bytes | str) -> AudioDecoder:
        return AudioDecoder(source, sample_rate=sample_rate, num_channels=num_channels)

    if isinstance(source, bytes):
        return decoder(source)

    if isinstance(source, str):
        return run_on_local_path(source, decoder)

    raise TypeError(f"不支持的音频输入类型：{type(source)}")


def encode_audio(audio: AudioDecoder | dict[str, Any], as_base64: bool = False) -> bytes | str:
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
        sf.write(buffer, array.T, sample_rate, format="WAV")
        wav_bytes = buffer.getvalue()
    # Step 3: Return as base64 or raw bytes
    if as_base64:
        return base64.b64encode(wav_bytes).decode("utf-8")
    return wav_bytes
