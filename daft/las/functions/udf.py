# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import daft
from daft.las.functions.types import NUM_CPUS, NUM_GPUS

if TYPE_CHECKING:
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
    init_args = construct_args or {}
    if num_gpus is not None:
        init_args.update({NUM_GPUS: num_gpus})
    if num_cpus is not None:
        init_args.update({NUM_CPUS: num_cpus})

    return daft.udf(
        return_dtype=daft.DataType.from_arrow_type(operator.__return_column_type__()),
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory_bytes=memory_bytes,
        batch_size=batch_size,
        concurrency=concurrency,
    )(operator).with_init_args(**init_args)
