# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def normalize_local_path(path: str) -> Path:
    return Path(path[7:]) if path.startswith("file://") else Path(path)


def generate_temp_file(path: str, suffix: str | None = None) -> str:
    """Generate a temporary file path with given path and suffix.

    The trailing `/` in path will be ignored.

    Parameters:
    ----------
    path : str
        The raw path.
    suffix : str, optional
        The given suffix will be as a suffix in the temp filename.
        The current timestamp will be used as part suffix in the temp filename if suffix is not set.

    Returns:
    -------
        The temp file path, the temp file is under the same parent dir with the given path.

    Examples:
    --------
    >>> temp_file = generate_temp_file("/a/b/c")
    >>> assert temp_file == f'/a/b/.c.temp-{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}'
    >>>
    >>> temp_file = generate_temp_file("/a/b/c", suffix="custom-suffix")
    >>> assert temp_file == "/a/b/.c.temp-custom-suffix"
    >>>
    >>> temp_file = generate_temp_file("a/b/c", suffix="custom-suffix")
    >>> assert temp_file == "a/b/.c.temp-custom-suffix"
    >>>
    >>> temp_file = generate_temp_file("file:///a/b//c", suffix="custom-suffix")
    >>> assert temp_file == "file:///a/b/.c.temp-custom-suffix"
    >>>
    >>> temp_file = generate_temp_file("/a/b/", suffix="custom-suffix")
    >>> assert temp_file == "/a/.b.temp-custom-suffix"
    >>>
    >>> temp_file = generate_temp_file("s3://bucket/a/b/c", suffix="custom-suffix")
    >>> assert temp_file == "s3://bucket/a/b/.c.temp-suffix"

    """
    path = path.rstrip("/")
    filename = path.rsplit("/", maxsplit=1)[-1]

    prefix = path[: len(path) - len(filename)]

    suffix = suffix or f'temp-{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}'
    return f"{prefix}.{filename}.{suffix}"
