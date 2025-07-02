# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


from daft.las.io import download_file, exists, upload_file


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


def path_to_base64(path: str) -> str:
    with Path(path).open(mode="rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


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
        temp_file_path = str(Path(temp_dir, Path(path).name))
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
