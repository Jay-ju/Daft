# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

from daft.las.io.factory import LasIO, register_io_client
from daft.las.io.utils import generate_temp_file, normalize_local_path

logger = logging.getLogger(__name__)


@register_io_client(scheme="http")
class HttpIO(LasIO):
    """The HttpIO provides downloading file from remote services."""

    def __init__(
        self, headers: dict[str, str] | None = None, max_retries: int = 3, backoff: int = 1, **kwargs: Any
    ) -> None:
        self.headers = headers
        self.max_retries = max_retries
        self.backoff = backoff

    @classmethod
    def scheme(cls) -> str:
        return "http"

    def mkdirs(self, path: str) -> None:
        raise NotImplementedError("mkdirs is not supported via http io")

    def exists(self, path: str) -> bool:
        raise NotImplementedError("exists is not supported via http io")

    def download_file(self, remote: str, local: str, overwrite: bool = True) -> None:
        local_file = normalize_local_path(local)
        if local_file.exists():
            if not overwrite:
                raise FileExistsError(f"The local file: {local} already exists.")
        else:
            parent = local_file.parent
            parent.mkdir(parents=True, exist_ok=True)

        temp_file = Path(generate_temp_file(str(local_file)))
        try:
            urlretrieve_with_retry(
                quote(remote, safe="/:?_=&%"), temp_file, self.headers, self.max_retries, self.backoff
            )

            # delete the existing file or dir after downloading file instead of deleting at
            # first to avoid loss the existing data as much as possible.
            if local_file.exists():
                if local_file.is_dir():
                    shutil.rmtree(local)
                else:
                    local_file.unlink()

            temp_file.rename(local_file)
        finally:
            temp_file.unlink(missing_ok=True)


def urlretrieve_with_retry(
    url: str,
    filename: str | Path | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    backoff: int = 1,
) -> tuple[str, str]:
    headers = headers or {}
    logger.info("urlretrieve_with_retry: url: %s, headers: %s", url, headers)
    req = urllib.request.Request(url, headers=headers)

    if filename is None:
        filename = os.path.basename(url) or "download"

    # Ensure we operate on a Path object for consistent behavior
    file_path = Path(filename)

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req) as response:
                # Write bytes to the path
                with open(file_path, "wb") as f:
                    f.write(response.read())
                return str(file_path), response.msg
        except urllib.error.HTTPError as e:
            if e.code >= 400 and attempt == max_retries:
                raise
        except urllib.error.URLError:
            if attempt == max_retries:
                raise
        if attempt < max_retries:
            time.sleep(backoff * (2**attempt))

    raise RuntimeError(f"Failed to download {url} to {file_path} after {max_retries} retries.")


@register_io_client(scheme="https")
class HttpsIO(HttpIO):
    """The HttpsIO provides downloading file from remote services."""

    @classmethod
    def scheme(cls) -> str:
        return "https"
