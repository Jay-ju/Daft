# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

T = TypeVar("T")


class MaxRetriesExceeded(Exception):
    """Exceed the maximum number or timeout of retries."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


logger = logging.getLogger(__name__)


class RetryPolicy:
    def __init__(
        self,
        max_retires: int = 3,
        max_retry_duration: float = 30.0,
        initial: float = 1,
        exp_base: float = 2,
        jitter: float = 1,
        max_wait: float = sys.maxsize / 2,
        retry_on_exceptions: Iterable[type[Exception]] | None = None,
        retry_condition: Callable[[T | None, Exception | None], bool] | None = None,
    ):
        self.max_retries = max_retires
        self.max_retry_duration = max_retry_duration

        self.initial = initial
        self.exp_base = exp_base
        self.jitter = jitter
        self.max_wait = max_wait

        self.retry_on_exceptions = retry_on_exceptions
        self.retry_condition = retry_condition

    def wait_exponential_jitter(self, attempt: int) -> float:
        jitter = random.uniform(0, self.jitter)
        try:
            exp = self.exp_base ** (attempt - 1)
            result = self.initial * exp + jitter
        except OverflowError:
            result = self.max_wait

        return max(0, min(result, self.max_wait))

    def with_max_retries(self, max_retries: int) -> RetryPolicy:
        self.max_retries = max_retries
        return self

    @staticmethod
    def should_retry(
        resp: T | None,
        ex: Exception | None,
        retry_on_exceptions: Iterable[type[Exception]] | None = None,
        retry_condition: Callable[[T | None, Exception | None], bool] | None = None,
    ) -> bool:
        # 1. Retry if the exception is retryable.
        if ex is not None:
            return retry_on_exceptions is not None and any(isinstance(ex, e) for e in retry_on_exceptions)

        # 2. Retry if match the customized retry condition.
        if retry_condition is not None and retry_condition(resp, ex):
            return True

        return False

    def retry_on(
        self,
        func: Callable[[int], T],
        max_retries: int | None = None,
        max_retry_duration: float | None = None,
        retry_on_exceptions: Iterable[type[Exception]] | None = None,
        retry_condition: Callable[[T | None, Exception | None], bool] | None = None,
    ) -> T:
        max_retries = max_retries or self.max_retries
        max_retry_duration = max_retry_duration or self.max_retry_duration
        retry_on_exceptions = retry_on_exceptions or self.retry_on_exceptions
        retry_condition = retry_condition or self.retry_condition  # type: ignore[assignment]

        start_time = datetime.now()
        last_exception = None
        for attempt in range(max_retries + 1):
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() >= max_retry_duration and attempt > 0:
                raise MaxRetriesExceeded(f"Retry duration exceeded: {max_retry_duration}s", last_exception)

            if attempt > 0:
                delay = self.wait_exponential_jitter(attempt)
                if delay > 0:
                    time.sleep(delay)

            try:
                response = func(attempt)

                # Note: should not return response even through response status code is 2xx,
                # because some responses might contain retryable errors
                if self.should_retry(response, None, retry_on_exceptions, retry_condition) and attempt < max_retries:
                    continue

                return response
            except Exception as exc:
                if self.should_retry(None, exc, retry_on_exceptions, retry_condition) and attempt < max_retries:
                    logger.debug("Retry the failed request. detail: %s", str(exc))
                    last_exception = exc
                    continue

                raise

        if last_exception:
            raise MaxRetriesExceeded(f"Retry times exceeded: {max_retries}", last_exception) from last_exception

        raise RuntimeError("Unexpected error in retry logic.")

    async def async_retry_on(
        self,
        func: Callable[[int], Awaitable[T]],
        max_retries: int | None = None,
        max_retry_duration: float | None = None,
        retry_on_exceptions: Iterable[type[Exception]] | None = None,
        retry_condition: Callable[[T | None, Exception | None], bool] | None = None,
    ) -> T:
        max_retries = max_retries or self.max_retries
        max_retry_duration = max_retry_duration or self.max_retry_duration
        retry_on_exceptions = retry_on_exceptions or self.retry_on_exceptions
        retry_condition = retry_condition or self.retry_condition  # type: ignore[assignment]

        start_time = datetime.now()
        last_exception = None
        for attempt in range(max_retries + 1):
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() >= max_retry_duration and attempt > 0:
                raise MaxRetriesExceeded(f"Retry duration exceeded: {max_retry_duration}s", last_exception)

            if attempt > 0:
                delay = self.wait_exponential_jitter(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

            try:
                response = await func(attempt)

                # Note: should not return response even through response status code is 2xx,
                # because some responses might contain retryable errors
                if self.should_retry(response, None, retry_on_exceptions, retry_condition) and attempt < max_retries:
                    continue

                return response
            except Exception as exc:
                if self.should_retry(None, exc, retry_on_exceptions, retry_condition) and attempt < max_retries:
                    logger.debug("Retry the failed request. detail: %s", str(exc))
                    last_exception = exc
                    continue

                raise

        if last_exception:
            raise MaxRetriesExceeded(f"Retry times exceeded: {max_retries}", last_exception) from last_exception

        raise RuntimeError("Unexpected error in retry logic.")
