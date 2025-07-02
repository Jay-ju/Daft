# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)


from PIL import Image  # noqa: TID253


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert a base64 string to a PIL image.

    Args:
        base64_str (str): Base64-encoded image string.

    Returns:
        Image.Image: The decoded image.
    """
    img_bytes = base64.b64decode(base64_str)
    img_buffer = io.BytesIO(img_bytes)
    return Image.open(img_buffer)


def binary_to_image(binary_str: bytes) -> Image.Image:
    """Convert a binary string to a PIL image.

    Args:
        binary_str (bytes): Image data in bytes.

    Returns:
        Image.Image: The decoded image.
    """
    img_buffer = io.BytesIO(binary_str)
    return Image.open(img_buffer)
