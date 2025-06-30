# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import daft

if TYPE_CHECKING:
    from daft.dependencies import pa
    from daft.las.functions.types import Operator
    from daft.udf import UDF


def las_udf(
    operator: type[Operator],
    construct_args: dict[str, Any] | None = None,
    num_cpus: float | None = None,
    num_gpus: float | None = None,
    memory_bytes: int | None = None,
    batch_size: int | None = None,
    concurrency: int | None = None,
) -> UDF:
    class Wrapper:
        def __init__(self) -> None:
            init_args = construct_args or {}
            if num_gpus is not None:
                init_args.update({"gpu_num": num_gpus})
            if num_cpus is not None:
                init_args.update({"num_cpus": num_cpus})

            self.delegate = operator(**init_args)

        def __call__(self, *columns: pa.Array) -> pa.Array:
            if len(columns) == 0:
                raise ValueError("The input data is empty")

            return self.delegate(*columns)

    return daft.udf(
        return_dtype=operator.__return_column_type__(),
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory_bytes=memory_bytes,
        batch_size=batch_size,
        concurrency=concurrency,
    )(Wrapper)
