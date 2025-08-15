# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.dependencies import pa
from daft.las.functions import Operator
from daft.las.functions.utils.common_utils import load_file


class LoadFileBytes(Operator):
    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.binary()

    def transform(self, file_urls: pa.Array) -> pa.Array:
        byte_array = [load_file(url.as_py(), as_base64=False) for url in file_urls]
        return pa.array(byte_array, type=pa.binary())


class LoadFileBase64(Operator):
    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()

    def transform(self, file_urls: pa.Array) -> pa.Array:
        byte_array = [load_file(url.as_py(), as_base64=True) for url in file_urls]
        return pa.array(byte_array, type=pa.string())
