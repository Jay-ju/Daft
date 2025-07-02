# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from daft.las.io.factory import LasIO, register_io_client
from daft.las.io.utils import normalize_local_path


@register_io_client(scheme="file")
class LocalIO(LasIO):
    """The LocalIO provides uploading and downloading file within local filesystem."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    @classmethod
    def scheme(cls) -> str:
        return "file"

    def mkdirs(self, path: str) -> None:
        if not path:
            raise ValueError("The path cannot be empty")

        normalize_local_path(path).mkdir(parents=True, exist_ok=True)

    def rm(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    def exists(self, path: str) -> bool:
        if not path:
            raise ValueError("The path cannot be empty")

        return normalize_local_path(path).exists()

    def file_size(self, path: str) -> int:
        if not path:
            raise ValueError("The path cannot be empty")

        return normalize_local_path(path).stat().st_size

    def _copy_file(self, src: str, dst: str, overwrite: bool = True) -> None:
        if src == dst:
            return

        if not src:
            raise ValueError("The src cannot be empty")
        if not dst:
            raise ValueError("The dst cannot be empty")

        src_file = normalize_local_path(src)
        dst_file = normalize_local_path(dst)
        if not src_file.exists():
            raise FileNotFoundError(f"The source file: {src} is not found.")
        if not src_file.is_file():
            raise FileNotFoundError(f"The source file: {src} is not a file.")

        if dst_file.exists():
            if not overwrite:
                raise FileExistsError(f"The destination file: {dst} already exists.")
            if dst_file.is_dir():
                shutil.rmtree(dst)
            else:
                dst_file.unlink()
        else:
            # Create parent dir if not found.
            parent = dst_file.parent
            parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(str(src_file), str(dst_file))

    def download_file(self, remote: str, local: str, overwrite: bool = True) -> None:
        self._copy_file(remote, local, overwrite)

    def upload_file(self, local: str, remote: str, overwrite: bool = True) -> None:
        self._copy_file(local, remote, overwrite)
