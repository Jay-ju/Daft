# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, TypeVar

T = TypeVar("T")

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
    use_process: bool = False,
) -> UDF:
    if hasattr(operator, "transform"):
        operator.__call__ = overwrite_method_signature(operator.__call__, operator.transform)  # type: ignore[method-assign]

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
        use_process=use_process
    )(operator).with_init_args(**init_args)


def overwrite_method_signature(target_func: Callable[..., Any], source_func: Callable[..., Any]) -> Callable[..., T]:
    source_sig = inspect.signature(source_func)

    @functools.wraps(target_func)
    def wrapper(*args: Any, **kwargs: Any) -> T:  # type: ignore[type-var]
        return target_func(*args, **kwargs)

    wrapper.__signature__ = source_sig  # type: ignore[attr-defined]
    wrapper.__annotations__ = source_func.__annotations__.copy()
    wrapper.__doc__ = target_func.__doc__ or source_func.__doc__

    return wrapper
