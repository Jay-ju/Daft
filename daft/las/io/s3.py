# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from daft.las.io.factory import LasIO, register_io_client
from daft.las.io.tos import TOSConfig
from daft.las.utils import not_blank

if TYPE_CHECKING:
    from daft.daft import S3Config


@register_io_client(scheme="s3")
class S3IO(LasIO):
    _io: LasIO

    def __init__(self, config: S3Config | None = None) -> None:
        if bool(os.getenv("CONVERT_S3_TO_TOS", True)):
            self._io = S3OnTosIO(config=config)
        else:
            self._io = S3LikeIO(config=config)

    @classmethod
    def scheme(cls) -> str:
        return "s3"

    def mkdirs(self, path: str) -> None:
        self._io.mkdirs(path)

    def rm(self, path: str) -> None:
        self._io.rm(path)

    def exists(self, path: str) -> bool:
        return self._io.exists(path)

    def file_size(self, path: str) -> int:
        return self._io.file_size(path)

    def download_file(self, remote: str, local: str, overwrite: bool = True) -> None:
        self._io.download_file(remote, local, overwrite)

    def upload_file(self, local: str, remote: str, overwrite: bool = True) -> None:
        self._io.upload_file(local, remote, overwrite)


class S3LikeIO(LasIO):
    def __init__(self, config: S3Config | None = None) -> None:
        raise NotImplementedError()


class S3OnTosIO(LasIO):
    def __init__(self, config: S3Config | None = None) -> None:
        from daft.las.io import TosIO

        tos_config = TOSConfig.from_s3_config(config) if config and not_blank(config.endpoint_url) else None
        self._io = TosIO(config=tos_config)

    @staticmethod
    def _convert_path(path: str) -> str:
        return path.replace("s3://", "tos://")

    def mkdirs(self, path: str) -> None:
        self._io.mkdirs(self._convert_path(path))

    def rm(self, path: str) -> None:
        self._io.rm(self._convert_path(path))

    def exists(self, path: str) -> bool:
        return self._io.exists(self._convert_path(path))

    def file_size(self, path: str) -> int:
        return self._io.file_size(self._convert_path(path))

    def download_file(self, remote: str, local: str, overwrite: bool = True) -> None:
        self._io.download_file(self._convert_path(remote), local, overwrite)

    def upload_file(self, local: str, remote: str, overwrite: bool = True) -> None:
        self._io.upload_file(local, self._convert_path(remote), overwrite)
