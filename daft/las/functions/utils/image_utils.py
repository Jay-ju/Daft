# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import io
import logging
from io import BytesIO
from typing import cast

import numpy as np  # noqa: TID253
import torch
from PIL import Image, ImageOps  # noqa: TID253
from torchvision.transforms import ToPILImage

from daft.las.functions.utils.common_utils import load_file, run_on_local_path

logger = logging.getLogger(__name__)


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert a base64 string to a PIL image.

    Args:
        base64_str: Base64-encoded image string.

    Returns:
        Image.Image: The decoded image.
    """
    img_bytes = base64.b64decode(base64_str)
    img_buffer = io.BytesIO(img_bytes)
    return Image.open(img_buffer)


def binary_to_image(binary_str: bytes) -> Image.Image:
    """Convert a binary string to a PIL image.

    Args:
        binary_str: Image data in bytes.

    Returns:
        Image.Image: The decoded image.
    """
    img_buffer = io.BytesIO(binary_str)
    return Image.open(img_buffer)


def url_to_image(url: str) -> Image.Image:
    """Convert a URL/path to a PIL Image object.

    Supports:
    - Local file paths
    - HTTP/HTTPS URLs
    - S3/TOS paths

    Args:
        url: Input resource locator string. Can be:
        - Local path (e.g. '/path/to/image.jpg')
        - Web URL (e.g. 'http://example.com/image.png')
        - S3 path (e.g. 's3://bucket/key.jpg', 'tos://bucket/key.jpg')

    Returns:
        PIL.Image.Image: Loaded image object
    """
    return run_on_local_path(url, lambda url: Image.open(url))


def decode_image(
    source: bytes | str,
    image_type: str = "image_base64",
) -> Image.Image:
    """Decodes image data from various input formats.

    Supports three encoding types:
    - image_base64: Base64 encoded string
    - image_binary: Raw binary data
    - image_url: URL/path to image file

    Args:
        source: Input data containing the image. Can be:
            - Base64 encoded string (for image_base64)
            - Bytes object (for image_binary)
            - URL/path string (for image_url)
        image_type: Specifies the encoding format of the source data.
            Supported values: 'image_base64', 'image_binary', 'image_url'
            Default: 'image_base64'

    Returns:
        PIL.Image.Image: Decoded image object

    Raises:
        ValueError: For unsupported image_type or invalid source data
        IOError: For file access/network errors (when using image_url)
    """
    if image_type == "image_base64":
        if not isinstance(source, str):
            raise ValueError("For image_base64, source must be str (base64).")
        return base64_to_image(source)
    elif image_type == "image_binary":
        if not isinstance(source, bytes):
            raise ValueError("For image_binary, source must be bytes.")
        return binary_to_image(source)
    elif image_type == "image_url":
        if not isinstance(source, str):
            raise ValueError("For image_url, source must be str (URL or path).")
        return url_to_image(source)
    else:
        raise ValueError(f"Invalid image_type: {image_type}")


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image object to a base64 string.

    Parameters
    ----------
        image (PIL.Image.Image): The image object to encode

    Returns:
    -------
        str: The encoded base64 string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_binary(image: Image.Image) -> bytes:
    """Converts a PIL Image object to binary format.

    Args:
        image (Image.Image): PIL Image object to be converted

    Returns:
        bytes: PNG-encoded binary data
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG", quality=85)
    return buffered.getvalue()


def decode_image_pil(source: str | bytes, mode: str | None = None) -> Image.Image:
    """Decode image from local/remote path or raw bytes into a PIL Image.

    Supported sources:
      - Raw image bytes
      - Local file paths
      - HTTP/TOS/S3 URIs

    Automatically handles EXIF orientation correction and optional mode conversion.

    Args:
        source: Input image source, either bytes or a path-like string.
        mode: Optional image mode to convert to (e.g., "RGB", "L"). If None, keeps original.

    Returns:
        Decoded image as a PIL Image object.
    """
    raw_bytes = cast("bytes", load_file(source))
    image = Image.open(BytesIO(raw_bytes))

    image.load()

    exif = image.getexif()
    if exif.get(Image.ExifTags.Base.Orientation) is not None:
        image = ImageOps.exif_transpose(image)

    if mode and image.mode != mode:
        image = image.convert(mode)

    return image


def encode_image(image: Image.Image | torch.Tensor | np.ndarray, as_base64: bool = False) -> bytes | str:
    # Step 1: Convert to PIL Image
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, (torch.Tensor, np.ndarray)):
        pil_image = ToPILImage()(image)
    else:
        raise TypeError(f"Unsupported image source type: {type(image)}")

    # Step 2: Save to buffer
    with BytesIO() as buf:
        image_format = "PNG" if pil_image.mode in ["1", "L", "LA", "RGB", "RGBA"] else "TIFF"
        pil_image.save(buf, format=image_format)
        encoded = buf.getvalue()

    # Step 3: Return bytes or base64
    if as_base64:
        return base64.b64encode(encoded).decode("utf-8")
    return encoded
