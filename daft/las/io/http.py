# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

from daft.las.io.factory import LasIO, register_io_client
from daft.las.io.util import generate_temp_file, normalize_local_path


@register_io_client(scheme="http")
class HttpIO(LasIO):
    """The HttpIO provides downloading file from remote services."""

    def __init__(self, **kwargs: Any) -> None:
        pass

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
            # TODO consider retry the download operation via a http client.
            urllib.request.urlretrieve(quote(remote, safe="/:?="), str(temp_file))

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


@register_io_client(scheme="https")
class HttpsIO(HttpIO):
    """The HttpsIO provides downloading file from remote services."""

    @classmethod
    def scheme(cls) -> str:
        return "https"
