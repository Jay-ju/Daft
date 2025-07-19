# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Literal, cast

import av

if TYPE_CHECKING:
    from av.container.input import InputContainer
from torchcodec.decoders import VideoDecoder

from daft.las.functions.utils.common_utils import load_file


def decode_video(source: str | bytes, mode: str = "r") -> InputContainer:
    """Decode video from local/remote path or raw bytes into an PyAV InputContainer.

    Supported sources:
      - Raw video bytes
      - Local file paths
      - HTTP/TOS/S3 URIs

    Args:
        source: Input video source, either bytes or a path-like string.
        mode: PyAV open mode, typically 'r' for reading.

    Returns:
        Decoded video container as an av.container.input.InputContainer.
    """
    raw_bytes = cast("bytes", load_file(source))
    return av.open(BytesIO(raw_bytes), mode=mode)


def decode_video_torchcodec(
    source: str | bytes,
    stream_index: int | None = None,
    dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
    num_ffmpeg_threads: int = 1,
    device: str | None = "cpu",
    seek_mode: Literal["exact", "approximate"] = "exact",
) -> VideoDecoder:
    raw_bytes = load_file(source)
    return VideoDecoder(
        raw_bytes,
        stream_index=stream_index,
        dimension_order=dimension_order,
        num_ffmpeg_threads=num_ffmpeg_threads,
        device=device,
        seek_mode=seek_mode,
    )


def encode_video(video: VideoDecoder, as_base64: bool = False) -> bytes | str:
    raise NotImplementedError()
