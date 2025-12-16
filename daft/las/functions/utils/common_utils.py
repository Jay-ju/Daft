# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import hashlib
import itertools
import logging
import os
import random
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

from daft.las.infra.tos_client import TosClient
from daft.las.io import download_file, exists, upload_file

T = TypeVar("T")


def base64_to_byte(base64_str: str) -> bytes:
    """Converts a Base64 string (possibly with a data URI prefix) to bytes.

    Args:
        base64_str (str): Base64 string that may contain a prefix (e.g. data URI scheme).

    Returns:
        bytes: The decoded bytes.
    """
    if "," in base64_str:
        _header, base64_str = base64_str.split(",", 1)
    return base64.b64decode(base64_str)


def byte_to_base64(byte_data: bytes) -> str:
    return base64.b64encode(byte_data).decode("utf-8")


def path_to_base64(path: str) -> str:
    with Path(path).open(mode="rb") as file:
        return byte_to_base64(file.read())


def path_to_byte(path: str) -> bytes:
    with Path(path).open(mode="rb") as file:
        return file.read()


def is_local_path(path: str) -> bool:
    return path.startswith(("/", "file://"))


def run_on_local_path(path: str, func: Callable[[str], T]) -> T:
    """Runs a callable on a local or remote file path.

    Args:
        path (str): The file path (local or remote).
        func (Callable[[str], T]): Function to run on the file.

    Returns:
        T: The result of the function.

    Raises:
        ValueError: If the path is empty.
        FileNotFoundError: If the local path does not exist.
        OtherError: If any other IO error occurs.
    """
    if not path:
        raise ValueError("Path cannot be empty")

    if is_local_path(path):
        if Path(path).exists():
            return func(path)
        raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory() as temp_dir:
        path_name = Path(path).name
        if len(path_name) >= 85:
            path_name = hashlib.md5(path_name.encode()).hexdigest()
        temp_file_path = str(Path(temp_dir, path_name))
        download_file(path, temp_file_path)
        return func(temp_file_path)


def upload_folder(local: str, uri: str, overwrite: bool = True, **kwargs: Any) -> None:
    """Uploads a local folder to a remote destination recursively.

    Args:
        local (str): Local folder path.
        uri (str): Remote destination folder.
        overwrite (bool, optional): Overwrite existing files. Defaults to True.
        **kwargs: Additional arguments passed through to upload_file().

    Raises:
        FileNotFoundError: If the local folder does not exist or is not a directory.
        OtherError: If any other IO error occurs.
    """
    local_path = Path(local)
    if not local_path.is_dir():
        raise FileNotFoundError(f"Local folder {local} not found or is not a directory.")

    for root, _, files in os.walk(local):
        for file in files:
            local_file_path = Path(root) / file
            relative_path = local_file_path.relative_to(local_path)
            remote_file_path = f"{uri.rstrip('/')}/{relative_path.as_posix()}"

            if overwrite:
                upload_file(str(local_file_path), remote_file_path, overwrite=True, **kwargs)
            else:
                if not exists(remote_file_path, **kwargs):
                    upload_file(str(local_file_path), remote_file_path, overwrite=False, **kwargs)


def encode_images_to_base64(directory: str) -> dict[str, str]:
    """Encodes all images in a directory to Base64 strings.

    Args:
        directory (str): Directory containing images.

    Returns:
        dict[str, str]: Mapping from filename to Base64-encoded string.
    """
    encoded_images: dict[str, str] = {}

    for file_path in Path(directory).iterdir():
        filename = file_path.name
        if file_path.is_file() and str(filename).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            with Path(file_path).open("rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                encoded_images[str(filename)] = encoded_string

    return encoded_images


def load_file(source: str | bytes, as_base64: bool = False) -> bytes | str:
    """Load binary data from local/remote path into memory as bytes or base64.

    Supported sources:
      - Raw bytes (direct return)
      - Local file paths
      - HTTP/TOS/S3 URIs

    Args:
        source: Input data source, either bytes or a path-like string.
        as_base64: Whether to return the content as a base64-encoded string.

    Returns:
        Raw bytes or base64-encoded string, depending on as_base64.
    """
    if isinstance(source, bytes):
        return base64.b64encode(source).decode("utf-8") if as_base64 else source

    if isinstance(source, str):

        def reader(path: str) -> bytes:
            return Path(path).read_bytes()

        try:
            data = run_on_local_path(source, reader)
            return base64.b64encode(data).decode("utf-8") if as_base64 else data
        except Exception as e:
            raise ValueError(f"Failed to load from source: {source}") from e

    raise TypeError(f"Unsupported input type: {type(source)}")


def base64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data)


def save_bytes_to_file(binary_data: bytes, save_path: str) -> None:
    """Save binary data (image, video, etc.) to a file.

    Args:
        binary_data (bytes): The binary data to save.
        save_path (str): The target file path including name and extension.
    """
    with Path(save_path).open("wb") as f:
        f.write(binary_data)


def save_file_to_local(content: str | bytes, content_type: str, directory: str, file_name: str) -> str:
    if "url" in content_type:
        if content and isinstance(content, str) and content.startswith(("tos://", "s3://", "https://", "http://")):
            tmp_file_name = str(Path(directory) / file_name)
        elif isinstance(content, str):
            tmp_file_name = content
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        download_file(content, tmp_file_name)
    else:
        tmp_file_name = str(Path(directory) / file_name)
        if "binary" in content_type and isinstance(content, bytes):
            content_binary = content
        elif "base64" in content_type and isinstance(content, str):
            content_binary = base64_to_byte(content)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        with Path(tmp_file_name).open("wb") as f:
            f.write(content_binary)
    return tmp_file_name


def pre_sign_url_for_tos(url: str, expires: int | None = None) -> str:
    """Pre-signs a URL for TOS.

    Args:
        url (str): The original URL.
        expires (int, optional): Expiration time in seconds. Defaults to 3600 * 24. Set TOS_PRE_SIGN_URL_EXPIRES to override.

    Returns:
        str: The pre-signed URL.
    """
    return TosClient().pre_sign_url(url, expires)


def _init_tracking_logger() -> logging.Logger:
    logging_level_str = os.getenv("LOG_LEVEL_FOR_USAGE_TRACKING", "WARN")
    logging_level = logging.getLevelName(logging_level_str.upper())
    if not isinstance(logging_level, int):
        raise ValueError(f"Invalid settings of LOG_LEVEL_FOR_USAGE_TRACKING: {logging_level_str}")

    usage_tracking_logger = logging.getLogger("UsageTracking")
    usage_tracking_logger.setLevel(logging_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
    handler.setFormatter(formatter)
    usage_tracking_logger.addHandler(handler)
    return usage_tracking_logger


usage_tracking_logger = _init_tracking_logger()


def tracking_usage(op: str, model_service_or_lib: str | None = None) -> None:
    usage_tracking_logger.info("Operator calling info: {op: %s, model_or_lib: %s}", op, model_service_or_lib)


def generate_filename_base_input(src_data: str, src_type: str, file_type: str, batch_idx: int, idx: int) -> str:
    if "url" in src_type and src_data and src_data.startswith(("tos://", "s3://")):
        file_name = f"{src_type.split('_')[0]}_{batch_idx}_{idx}.{src_data.split('.')[-1]}"
    else:
        file_name = f"{int(time.time())!s}_{batch_idx}_{idx}.{file_type}"

    return file_name


def generate_filename_prefix(
    src_type: str, idx: int, original_content: list[str], original_content_name: list[str], suffix: str = ""
) -> str:
    try:
        if original_content_name and len(original_content_name) == len(original_content):
            name = original_content_name[idx]
        elif "url" in src_type:
            parsed = urlparse(original_content[idx])
            filename = Path(parsed.path).name or f"binary_{uuid.uuid4().hex}"
            filename = filename.split("?")[0]
            p = Path(filename)
            if p.suffix:
                name = p.stem
            else:
                name = filename
        else:
            name = f"{int(time.time())!s}_{(random.randint(1, 1000000))!s}"
    except Exception:
        # 文件名生成兜底逻辑
        name = f"{int(time.time())!s}_{(random.randint(1, 1000000))!s}"
    return f"{name}{suffix}"


class FastWriteCounter:
    """Thread-safe counter for generating sequential indices.

    Used for tracking batch processing progress in parallel operations.
    """

    def __init__(self, init: int = 0, step: int = 1) -> None:
        self._number_of_read = 0
        self._step = step
        self._counter = itertools.count(init, step)
        self._lock = threading.Lock()

    def increment(self) -> None:
        next(self._counter)

    @property
    def value(self) -> int:
        with self._lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += self._step
        return value


def get_logger(name: str) -> logging.Logger:
    """Create or get a logger with standardized formatting.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
