# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any

from daft import Series

if TYPE_CHECKING:
    from logging import Logger

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


class AsyncOperatorStats:
    def __init__(self, logger: Logger) -> None:
        self._lock = asyncio.Lock()
        self._stats: Counter[str] = Counter()
        self.logger = logger

    async def _inc(self, key: str, value: int = 1) -> None:
        async with self._lock:
            self._stats[key] += value

    async def log_accept(self, num: int = 1) -> None:
        await self._inc("accepted", num)

    async def log_submit(self) -> None:
        await self._inc("submitted")

    async def log_succeed(self) -> None:
        await self._inc("succeed")

    async def log_failed(self) -> None:
        await self._inc("failed")

    async def reset(self) -> None:
        async with self._lock:
            self._stats.clear()

    async def log_process(self) -> None:
        async with self._lock:
            accepted = self._stats["accepted"]
            submitted = self._stats["submitted"]
            succeed = self._stats["succeed"]
            failed = self._stats["failed"]
            finished = succeed + failed
            running = submitted - finished
            pending = accepted - submitted

        self.logger.info(
            "%d running, %d pending, accepted/submitted/finished/succeed/failed: %d/%d/%d/%d/%d",
            running,
            pending,
            accepted,
            submitted,
            finished,
            succeed,
            failed,
        )


class EventLooper:
    def __init__(self) -> None:
        self._local = threading.local()

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        if not hasattr(self._local, "loop"):
            self._local.loop = asyncio.new_event_loop()
        return self._local.loop

    def run(self, func: Any) -> Any:
        loop = self._get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(func, loop).result()
        return loop.run_until_complete(func)
