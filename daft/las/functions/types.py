# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from daft import Series

if TYPE_CHECKING:
    import pyarrow as pa


NUM_CPUS = "num_cpus"
NUM_GPUS = "num_gpus"
DEFAULT_NUM_GPUS = 1


class Operator(ABC):
    """An operator that process the input data."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        num_gpus = kwargs.get(NUM_GPUS, DEFAULT_NUM_GPUS)
        self.use_gpu = False
        if num_gpus > 0:
            self.use_gpu = True
        import math

        self.cuda_device_count = math.ceil(num_gpus)

    def __call__(self, *args: Any, **kwargs: Any) -> pa.Array:
        _args = [arg.to_arrow() if isinstance(arg, Series) else arg for arg in args]
        _kwargs = {key: arg.to_arrow() if isinstance(arg, Series) else arg for key, arg in kwargs.items()}
        return self.transform(*_args, **_kwargs)

    def transform(self, *args: Any, **kwargs: Any) -> pa.Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def __return_column_type__() -> pa.DataType: ...
