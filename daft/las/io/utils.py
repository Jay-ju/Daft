# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def normalize_local_path(path: str) -> Path:
    return Path(path[7:]) if path.startswith("file://") else Path(path)


def generate_temp_file(path: str, suffix: str | None = None) -> str:
    """Generate a temporary file path with the given path and suffix.

    This function creates a temporary file path by modifying the original path:
    1. Removes any trailing slash
    2. Prefixes the filename with a dot (.)
    3. Adds the string ".temp" before the suffix
    4. Appends the custom suffix or a timestamp if no suffix is provided

    The resulting file is in the same directory as the original path.

    Args:
        path: The original file path (trailing slashes are ignored)
        suffix: Custom suffix to append to the filename. If not provided,
                a timestamp in the format YYYYMMDDHHMMSS will be used.

    Returns:
        The generated temporary file path.

    Examples:
        >>> generate_temp_file("/a/b/c")
        '/a/b/.c.temp-20230315124530'  # Actual timestamp will vary

        >>> generate_temp_file("/a/b/c", "custom-suffix")
        '/a/b/.c.temp-custom-suffix'

        >>> generate_temp_file("/a/b/", "suffix")
        '/a/.b.temp-suffix'

        >>> generate_temp_file("s3://bucket/a/b/c", "suffix")
        's3://bucket/a/b/.c.temp-suffix'

    Implementation details:
        - For files in root directories: /file → /.file.temp-{suffix}
        - Handles all types of paths: local, s3, gs, etc.
    """
    path = path.rstrip("/")
    filename = path.rsplit("/", maxsplit=1)[-1]

    prefix = path[: len(path) - len(filename)]

    suffix = suffix or f'temp-{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}'
    return f"{prefix}.{filename}.{suffix}"
