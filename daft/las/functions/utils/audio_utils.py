# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging

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
