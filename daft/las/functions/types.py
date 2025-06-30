# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from daft.datatype import DataTypeLike

NUM_GPUS = "num_gpus"
DEFAULT_NUM_GPUS = 1


class Operator(ABC):
    """An operator that process the input data."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        num_gpus = kwargs.get(NUM_GPUS, DEFAULT_NUM_GPUS)
        self.use_gpu = False
        if num_gpus > 0:
            self.use_gpu = True
        import math

        self.cuda_device_count = math.ceil(num_gpus)

    def __call__(self, *columns: pa.Array | pa.Table) -> pa.Array:
        return self.transform(*columns)

    def transform(self, *columns: pa.Array) -> pa.Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def __return_column_type__() -> DataTypeLike: ...
