from __future__ import annotations

import asyncio
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from logging import Logger

    import httpx

    from daft.las.infra.http.retry import RetryPolicy

from daft.las.functions import Operator
from daft.las.functions.types import AsyncOperatorStats, EventLooper
from daft.las.infra.http.auth import AnonymousProvider, ApiKeyAuthProvider, AuthProvider
from daft.las.infra.http.client import DEFAULT_RETRY_POLICY, AsyncHttpClient, default_retry_condition

DEFAULT_LAS_OPERATOR_ENDPOINT = "https://operator.las.cn-beijing.volces.com"


class HttpOperator(Operator, ABC):
    def __init__(
        self,
        base_url: str,
        auth_provider: AuthProvider = AnonymousProvider(),
        retry_config: RetryPolicy = DEFAULT_RETRY_POLICY,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.client = AsyncHttpClient(
            base_url=base_url,
            auth_provider=auth_provider,
            retry_config=retry_config,
        )

    async def send(self, method: str, path: str, payload: Any | None = None) -> httpx.Response:
        return await self.client.request(
            method=method,
            path=path,
            json=payload,
            retry_condition=self.retry_condition,
        )

    def retry_condition(self, resp: httpx.Response | None, ex: Exception | None) -> bool:
        return default_retry_condition(resp, ex)


class LasOperator(HttpOperator, ABC):
    def __init__(
        self,
        operator_id: str,
        operator_version: str = "v1",
        api_key: str | None = None,
        endpoint: str = DEFAULT_LAS_OPERATOR_ENDPOINT,
        version: str = "v1",
        max_retries: int = 3,
        **kwargs: Any,
    ):
        super().__init__(
            base_url=self._base_url(endpoint, version),
            auth_provider=self._auth_provider(api_key),
            retry_config=DEFAULT_RETRY_POLICY.with_max_retries(max_retries),
            **kwargs,
        )

        self.operator_id = operator_id
        self.operator_version = operator_version

    @classmethod
    def _base_url(cls, endpoint: str | None, version: str) -> str:
        endpoint = endpoint or os.getenv("LAS_SERVICE_ENDPOINT")
        if endpoint is None:
            raise ValueError(
                "Endpoint is required. Need to set endpoint implicitly or set LAS_SERVICE_ENDPOINT environment variable."
            )

        endpoint = endpoint.rstrip("/")
        if not endpoint.startswith("http"):
            raise ValueError(f"Endpoint: {endpoint} is invalid. only support http/https protocol.")

        return f"{endpoint}/api/{version}"

    @classmethod
    def _auth_provider(cls, api_key: str | None) -> AuthProvider:
        api_key = api_key or os.getenv("LAS_API_KEY")
        if not api_key:
            raise ValueError("api_key is missing: provided via parameter or set LAS_API_KEY environment variable")

        return ApiKeyAuthProvider(api_key=api_key)

    def _request_payload(self, data: Any | None = None) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "data": data,
        }

    async def process(self, data: Any | None = None) -> httpx.Response:
        return await self.send(method="POST", path="process", payload=self._request_payload(data))

    async def submit(self, data: Any | None = None) -> httpx.Response:
        return await self.send(method="POST", path="submit", payload=self._request_payload(data))

    async def poll(self, task_id: str) -> httpx.Response:
        return await self.send(
            method="POST",
            path="poll",
            payload={
                "operator_id": self.operator_id,
                "operator_version": self.operator_version,
                "task_id": task_id,
            },
        )


class LasPollOperator(LasOperator, ABC):
    def __init__(
        self,
        operator_id: str,
        operator_version: str = "v2",
        api_key: str | None = None,
        endpoint: str = DEFAULT_LAS_OPERATOR_ENDPOINT,
        version: str = "v1",
        max_retries: int = 5,
        num_coroutines: int = 20,
        max_polling_seconds: int = 7200,
        max_polling_num: int = 240,
        waiting_initial: float = 1,
        waiting_exp_base: float = 2,
        waiting_jitter: float = 1,
        waiting_max_seconds: float = 30,
        **kwargs: Any,
    ):
        """创建一个 Poller 实例，并发轮询任务状态与结果（Poll）.

        - 异步流程：携带 `task_id` 调用（POST /api/v1/poll）直至 `COMPLETED`/`FAILED`/`TIMEOUT`
        - 具体实现类需要实现下列方法：
            - `logger()`: 返回 Poller 实例的 Logger 实例。
            - `default_result()`: 返回默认的轮询结果。
            - `parse_response(resp: dict[str, Any]) -> dict[str, Any]`: 解析轮询响应，返回解析后的结果。
            - `__return_column_type__()`: 返回轮询结果的 PyArrow 数据类型，处理业务需要的字段之外，还需要额外声明`pa.field("failed_reason", pa.string())`。



        Args:
            operator_id: 算子 ID。
            operator_version: 算子版本，默认 "v2"。
            api_key: LAS 服务 API Key。若为空，将从环境变量 `LAS_API_KEY` 读取。
            endpoint: LAS 服务 API Endpoint。若为空，将读取环境变量 `LAS_SERVICE_ENDPOINT`。
            version: API 版本，默认 "v1"。
            max_retries: 单次轮询请求最大重试次数，默认 5。
            num_coroutines: 并发轮询的最大数量，默认 20。
            max_polling_seconds: 最大等待时长（秒）。若未设置，从环境变量 `LAS_MAX_WAIT_SECONDS` 读取，默认 7200。
            max_polling_num: 最大轮询次数，默认 240。
            waiting_initial: 轮询间隔（秒）。若未设置，从环境变量 `LAS_POLL_INTERVAL_SECONDS` 读取，默认 1。
            waiting_exp_base: 指数退避底数，默认 2。
            waiting_jitter: 退避抖动范围，默认 1。
            waiting_max_seconds: 单次最大等待上限（秒），默认 30。

        Raises:
            ValueError: 当 `api_key` 或 `endpoint` 缺失且无法从环境变量获取时抛出。

        """
        super().__init__(
            operator_id=operator_id,
            operator_version=operator_version,
            api_key=api_key,
            endpoint=endpoint,
            version=version,
            max_retries=max_retries,
            **kwargs,
        )

        self.semaphore = asyncio.Semaphore(num_coroutines)
        self.event_loop = EventLooper()

        self.logger = self.get_logger()
        self.stats = AsyncOperatorStats(self.logger)

        self.max_polling_seconds = max_polling_seconds
        self.max_polling_num = max_polling_num
        self.waiting_initial = (
            waiting_initial if waiting_initial is not None else float(os.getenv("LAS_POLL_INTERVAL_SECONDS", 1.0))
        )
        self.waiting_exp_base = waiting_exp_base
        self.waiting_jitter = waiting_jitter
        self.waiting_max_seconds = waiting_max_seconds

        self.default_result = self.get_default_result()

    @abstractmethod
    def get_logger(self) -> Logger:
        """返回 Poller 实例的 Logger 实例."""
        ...

    @abstractmethod
    def get_default_result(self) -> dict[str, str]:
        """返回默认的轮询结果."""
        ...

    @abstractmethod
    def parse_response(self, resp: dict[str, Any]) -> dict[str, Any]:
        """解析轮询响应，返回解析后的结果."""
        ...

    def batch_poll(self, task_ids: list[str], refs: list[str]) -> list[dict[str, Any]]:
        return self.event_loop.run(self.async_batch_poll(task_ids, refs))

    async def async_batch_poll(self, task_ids: list[str], refs: list[str]) -> list[dict[str, Any]]:
        await self.stats.log_accept(len(task_ids))

        async def _single_poll(task_id: str, ref: str) -> dict[str, Any]:
            async with self.semaphore:
                return await self.single_poll(task_id, ref)

        # log the process before current batch
        await self.stats.log_process()
        result = await asyncio.gather(*[_single_poll(task_id, ref) for task_id, ref in zip(task_ids, refs)])
        # log the process after current batch
        await self.stats.log_process()

        return result

    async def single_poll(self, task_id: str, ref: str) -> dict[str, str]:
        await self.stats.log_submit()

        if not task_id:
            await self.stats.log_failed()
            self.logger.warning("task id is empty for %s, skip.", ref)
            return {**self.default_result, "failed_reason": "SUBMIT_TASK_FAILED"}

        start = datetime.now()
        attempts = 0
        while True:
            try:
                # 1. send poll request
                resp = await self.poll(task_id)

                # 2. check http status code
                attempt_num = resp.headers.get("attempt_num", 0)
                http_status = resp.status_code
                if http_status != 200:
                    await self.stats.log_failed()

                    self.logger.warning(
                        "Failed to query task status for %s, task: %s, http status: %s, attempt: %s, detail: %s",
                        ref,
                        task_id,
                        resp.status_code,
                        attempt_num,
                        resp.text,
                    )

                    # try to parse the exact error message from response json
                    failed_reason = resp.text
                    try:
                        error_msg = resp.json().get("metadata", {}).get("error_msg", "")
                        if error_msg:
                            failed_reason = error_msg
                    except ValueError:
                        pass

                    return {**self.default_result, "failed_reason": failed_reason}

                result = resp.json()
                self.logger.debug("Polling response for %s, task: %s, resp: %s", ref, task_id, result)

                # 3. check task status
                meta = result.get("metadata", {})
                req_id = meta.get("request_id", "")
                task_status = meta.get("task_status", "UNKNOWN")
                if task_status == "COMPLETED":
                    await self.stats.log_succeed()
                    return self.parse_response(result.get("data", {}))
                elif task_status in ["FAILED", "TIMEOUT"]:
                    await self.stats.log_failed()
                    self.logger.warning(
                        "The task %s of %s is failed, req id: %s, detail: %s", task_id, ref, req_id, meta
                    )
                    return {**self.default_result, "failed_reason": meta.get("error_msg", task_status)}
                elif task_status in ["ACCEPTED", "PENDING", "RUNNING"]:
                    waiting = self.wait_exponential_jitter(attempts + 1)
                    self.logger.debug(
                        "Task: %s doesn't finished, ref: %s, status: %s, sleep %fs",
                        task_id,
                        ref,
                        task_status,
                        waiting,
                    )
                    await asyncio.sleep(waiting)
                else:
                    await self.stats.log_failed()

                    self.logger.warning(
                        "Unexpected task status: %s, task: %s, ref: %s, req id: %s, skip.",
                        task_status,
                        task_id,
                        ref,
                        req_id,
                    )
                    return {**self.default_result, "failed_reason": "UNEXPECTED_TASK_STATUS"}

                attempts += 1
                cost = (datetime.now() - start).total_seconds()
                if cost > self.max_polling_seconds:
                    await self.stats.log_failed()
                    self.logger.warning(
                        "Elapsed max query time: %ss, ref: %s, task: %s.", self.max_polling_seconds, ref, task_id
                    )
                    return {**self.default_result, "failed_reason": "ELAPSED_POLLING_TIME"}

                if attempts > self.max_polling_num:
                    await self.stats.log_failed()

                    self.logger.warning(
                        "Elapsed max query number: %s, ref: %s, task: %s.", self.max_polling_num, ref, task_id
                    )
                    return {**self.default_result, "failed_reason": "ELAPSED_POLLING_NUM"}

                poll_stat = {"task_id": task_id, "ref": ref, "acc_cost": cost, "acc_num": attempts}
                self.logger.info("The polling stats: %s", poll_stat)
            except Exception as e:
                await self.stats.log_failed()

                self.logger.exception("Failed to query task status for %s, task: %s.", ref, task_id)
                return {**self.default_result, "failed_reason": str(e)}

    def wait_exponential_jitter(self, attempt: int) -> float:
        jitter = random.uniform(0, self.waiting_jitter)
        try:
            exp = self.waiting_exp_base ** (attempt - 1)
            result = self.waiting_initial * exp + jitter
        except OverflowError:
            result = self.waiting_max_seconds

        return max(0, min(result, self.waiting_max_seconds))
