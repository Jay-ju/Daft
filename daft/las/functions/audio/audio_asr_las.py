from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime
from typing import TYPE_CHECKING, Any

from daft.dependencies import pa
from daft.las.functions import Operator
from daft.las.functions.types import AsyncOperatorStats, EventLooper
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.infra.http.auth import ApiKeyAuthProvider
from daft.las.infra.http.client import DEFAULT_RETRY_POLICY, AsyncHttpClient, default_retry_condition

if TYPE_CHECKING:
    import httpx


class LasAsrSubmitter(Operator):
    """**语音识别模块 - 基于LAS ASR服务的录音转写解决方案**

    **核心功能**
    - 接入火山引擎LAS ASR接口
    - 支持自动断句、数字规整、说话人或通道分离（可选）
    - 并发处理多个音频文件，提供结构化 JSON 与可读文本两种输出
    - 适合转写最长2小时的录音文件，支持标点补全、智能断句、说话人分离等高级功能。

    **使用场景**
    - 智能会议纪要生成（推荐开启说话人识别）
    - 课后教学音频分析、学情回顾
    - 语音客服/外呼质检分析
    - 媒体节目、采访播客的离线字幕生成
    """  # noqa: D415

    def __init__(
        self,
        api_key: str,
        endpoint: str | None = None,
        version: str = "v1",
        operator_id: str = "las_asr",
        operator_version: str = "v1",
        num_coroutines: int = 20,
        max_retries: int = 3,
        enable_idempotent: bool = False,
        enable_punc: bool = False,
        enable_ddc: bool = False,
        enable_speaker_info: bool = False,
        enable_itn: bool = True,
        enable_channel_split: bool = False,
        enable_lid: bool = False,
        show_speech_rate: bool = False,
        show_volume: bool = False,
        **kwargs: Any,
    ) -> None:
        """创建一个LasAsrSubmitter实例，用于提交音频到LAS ASR 服务进行处理.

        Args:
            api_key: LAS服务 API Key。
            endpoint: LAS服务 API Endpoint。
            version: LAS服务 API Version，默认值: "v1"。
            operator_id: LAS ASR服务ID，默认值: "las_asr"。
            operator_version: LAS ASR服务版本，默认值: "v1"。
            num_coroutines: 单个实例并发处理音频的最大数量，默认值: 20。
            max_retries: 单次API请求最大重试次数，默认值：3。
            enable_idempotent: 用于判断是否重试处理数据，为True时，每次都会重新处理，默认值：False。
            enable_punc: 文本标点，将原始语音输出转换为带标点的形式，以提高文本的可读性，默认值：False。
            enable_ddc: 语义顺滑，旨在提高自动语音识别（ASR）结果的文本可读性和流畅性。这项技术通过删除或修改ASR结果中的不流畅部分，如停顿词、语气词、语义重复词等，使得文本更加易于阅读和理解，默认值：False。
            enable_speaker_info: 语音角色信息，开启后可返回说话人的信息，10人以内，效果较好，默认值：False。
            enable_itn: 文本规范化，将原始语音输出转换为书面形式，以提高文本的可读性，默认值：True。
            enable_channel_split: 语音分轨，开启后会在返回结果中使用channel_id标记，1为左声道，2为右声道，默认值：False。
            enable_lid: 语言识别，目前支持语种：中英文、上海话、闽南语，四川、陕西、粤语，开启后会在additions信息中使用lid_lang标记, 返回对应的语种标签，默认值：False。
            show_speech_rate: 语速信息，开启后会在分句additions信息中使用speech_rate标记，单位为 token/s，默认值：False。
            show_volume: 音量信息，开启后可在分句additions信息中使用volume标记，单位为 分贝，默认值：False。
        """
        super().__init__(**kwargs)

        if not api_key:
            raise ValueError(f"api_key: {api_key} is invalid.")

        # TODO set a default online endpoint
        endpoint = endpoint or os.getenv("LAS_SERVICE_ENDPOINT", "")
        if endpoint is None or not endpoint.startswith("http"):
            raise ValueError(f"endpoint: {endpoint} is invalid.")

        self.client = AsyncHttpClient(
            base_url=f"{endpoint}/api/{version}/",
            auth_provider=ApiKeyAuthProvider(api_key),
            retry_config=DEFAULT_RETRY_POLICY.with_max_retries(max_retries),
        )

        self.static_template: dict[str, Any] = {
            "operator_id": operator_id,
            "operator_version": operator_version,
            "idempotent_id": enable_idempotent,
            "data": {
                "enable_itn": enable_itn,
                "enable_punc": enable_punc,
                "enable_ddc": enable_ddc,
                "enable_speaker_info": enable_speaker_info,
                "enable_channel_split": enable_channel_split,
                "show_speech_rate": show_speech_rate,
                "show_volume": show_volume,
                "enable_lid": enable_lid,
            },
        }

        self.semaphore = asyncio.Semaphore(num_coroutines)
        self.logger = logging.getLogger(f"LasAsrSubmitter-{id(self)}")
        self.stats = AsyncOperatorStats(self.logger)
        self.event_loop = EventLooper()

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="openspeech")

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()

    def transform(self, audio_urls: pa.Array) -> pa.Array:
        results = self.event_loop.run(self.run(audio_urls.to_pylist()))
        return pa.array(results, type=self.__return_column_type__())

    async def run(self, audio_urls: list[str]) -> list[str]:
        await self.stats.log_accept(len(audio_urls))

        async def _submit(url: str) -> str:
            async with self.semaphore:
                return await self.submit(url)

        # log the process before current batch
        await self.stats.log_process()
        result = await asyncio.gather(*[_submit(url) for url in audio_urls])
        # log the process after current batch
        await self.stats.log_process()

        return result

    @classmethod
    def _retry_condition(cls, resp: httpx.Response | None, ex: Exception | None) -> bool:
        if default_retry_condition(resp, ex):
            return True

        if resp is not None:
            meta = resp.json().get("metadata", {})
            # 2002: TIMEOUT_ERROR
            # 2003: SERVER_BUSY
            if meta and meta.get("business_code") in ["2002", "2003"]:
                return True

        return False

    async def submit(self, url: str) -> str:
        await self.stats.log_submit()

        if not url:
            await self.stats.log_failed()
            self.logger.warning("url is empty, skip")
            return ""

        if not url.startswith("http"):
            await self.stats.log_failed()
            self.logger.warning("url: %s is invalid, skip", url)
            return ""

        try:
            resp = await self.client.request(
                method="POST",
                path="submit",
                json={**self.static_template, "data": {**self.static_template["data"], "audio_url": url}},
                retry_condition=self._retry_condition,
            )

            attempt_num = resp.headers.get("attempt_num", 0)
            meta = resp.json().get("metadata", {})
            if resp.status_code != 200 or meta.get("business_code") != "0":
                await self.stats.log_failed()
                self.logger.warning(
                    "Failed to sumit url: %s, http status: %s, attempt: %s, detail: %s",
                    url,
                    resp.status_code,
                    attempt_num,
                    resp.text,
                )
                return ""

            await self.stats.log_succeed()
            return meta.get("task_id", "")
        except Exception:
            await self.stats.log_failed()
            self.logger.exception("Failed to sumit url: %s.", url)
            return ""


class LasAsrPoller(Operator):
    def __init__(
        self,
        api_key: str,
        endpoint: str | None = None,
        version: str = "v1",
        operator_id: str = "las_asr",
        operator_version: str = "v1",
        num_coroutines: int = 20,
        max_retries: int = 5,
        max_polling_seconds: int = 7200,
        max_polling_num: int = 240,
        waiting_initial: float = 1,
        waiting_exp_base: float = 2,
        waiting_jitter: float = 1,
        waiting_max_seconds: float = 30,
        **kwargs: Any,
    ):
        """创建一个LasAsrPoller实例，并发地轮询多个ASR任务的结果.

        Args:
            api_key: LAS服务 API Key。
            endpoint: LAS服务 API Endpoint。
            version: LAS服务 API Version，默认值: "v1"。
            operator_id: LAS ASR服务ID，默认值: "las_asr"。
            operator_version: LAS ASR服务版本，默认值: "v1"。
            num_coroutines: 轮询task状态的最大并发数，默认值: 20。
            max_retries: 单次轮询API请求的最大重试次数，默认值: 5。
            max_polling_seconds: 单个个task最大轮询时长，单位：秒，默认值: 7200。
            max_polling_num: 单个task最大轮询次数，默认值: 240。
            waiting_initial: 轮询初始等待时间，默认值: 1。
            waiting_exp_base: 轮询指数退避等待时间的底数，默认值: 2。
            waiting_jitter: 轮询随机退避等待时间的范围，默认值: 1。
            waiting_max_seconds: 单次最大等待时长，单位：秒，默认值: 30。
        """
        super().__init__(**kwargs)

        if not api_key:
            raise ValueError(f"api_key: {api_key} is invalid.")

        # TODO set a default online endpoint
        endpoint = endpoint or os.getenv("LAS_SERVICE_ENDPOINT", "")
        if endpoint is None or not endpoint.startswith("http"):
            raise ValueError(f"endpoint: {endpoint} is invalid.")

        self.client = AsyncHttpClient(
            base_url=f"{endpoint}/api/{version}/",
            auth_provider=ApiKeyAuthProvider(api_key),
            retry_config=DEFAULT_RETRY_POLICY.with_max_retries(max_retries),
        )

        self.static_template = {
            "operator_id": operator_id,
            "operator_version": operator_version,
        }

        self.semaphore = asyncio.Semaphore(num_coroutines)
        self.logger = logging.getLogger(f"LasAsrPoller-{id(self)}")
        self.stats = AsyncOperatorStats(self.logger)
        self.max_polling_seconds = max_polling_seconds
        self.max_polling_num = max_polling_num
        self.polling_stats: dict[str, Any] = {}
        self.waiting_initial = waiting_initial
        self.waiting_exp_base = waiting_exp_base
        self.waiting_jitter = waiting_jitter
        self.waiting_max_seconds = waiting_max_seconds
        self.default_result = {"asr_result_raw": "", "asr_result_text": ""}
        self.event_loop = EventLooper()

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("asr_result_raw", pa.string()),
                pa.field("asr_result_text", pa.string()),
                pa.field("failed_reason", pa.string()),
            ]
        )

    @classmethod
    def _retry_condition(cls, resp: httpx.Response | None, ex: Exception | None) -> bool:
        if default_retry_condition(resp, ex):
            return True

        if resp is not None:
            meta = resp.json().get("metadata", {})
            # 2002: TIMEOUT_ERROR
            # 2003: SERVER_BUSY
            if meta and meta.get("business_code") in ["2002", "2003"]:
                return True

        return False

    def transform(self, audios: pa.Array, tasks: pa.Array) -> pa.Array:
        results = self.event_loop.run(self.run(audios.to_pylist(), tasks.to_pylist()))
        return pa.array(results, type=self.__return_column_type__())

    async def run(self, audios: list[str], tasks: list[str]) -> list[dict[str, str]]:
        await self.stats.log_accept(len(audios))

        async def _poll(audio: str, task: str) -> dict[str, str]:
            async with self.semaphore:
                return await self.poll(audio, task)

        # log the process before current batch
        await self.stats.log_process()
        result = await asyncio.gather(*[_poll(url, task) for url, task in zip(audios, tasks)])
        # log the process after current batch
        await self.stats.log_process()

        return result

    async def poll(self, audio: str, task_id: str) -> dict[str, str]:
        def clean_polling_stats() -> None:
            self.polling_stats.pop(audio, None)

        await self.stats.log_submit()

        if not task_id:
            await self.stats.log_failed()
            self.logger.warning("task id is empty, audio: %s, skip.", audio)
            return {**self.default_result, "failed_reason": "SUBMIT_TASK_FAILED"}

        start = datetime.now()
        attempts = 0
        while True:
            try:
                resp = await self.client.request(
                    method="POST",
                    path="poll",
                    json={**self.static_template, "task_id": task_id},
                    retry_condition=self._retry_condition,
                )

                attempt_num = resp.headers.get("attempt_num", 0)
                if resp.status_code != 200:
                    await self.stats.log_failed()
                    clean_polling_stats()

                    self.logger.warning(
                        "Failed to query task status, audio: %s, task: %s, http status: %s, attempt: %s, detail: %s",
                        audio,
                        task_id,
                        resp.status_code,
                        attempt_num,
                        resp.text,
                    )
                    return {**self.default_result, "failed_reason": str(resp.status_code)}

                result = resp.json()
                meta = resp.json().get("metadata", {})
                if meta.get("business_code") != "0":
                    await self.stats.log_failed()
                    self.logger.warning(
                        "Failed to query task status, audio: %s, task: %s, http status: %s, attempt: %s, detail: %s",
                        audio,
                        task_id,
                        resp.status_code,
                        attempt_num,
                        resp.text,
                    )
                    return {
                        **self.default_result,
                        "failed_reason": str(meta.get("error_msg", "KNOWN_BUSINESS_CODE")),
                    }

                task_status = meta.get("task_status", "UNKNOWN")
                data = result.get("data", {})

                if task_status == "COMPLETED":
                    await self.stats.log_succeed()
                    clean_polling_stats()
                    return {
                        "asr_result_raw": data["asr_result_raw"],
                        "asr_result_text": data["asr_result_text"],
                        "failed_reason": "",
                    }
                elif task_status == "FAILED":
                    await self.stats.log_failed()
                    clean_polling_stats()
                    metadata = result.get("metadata", {})
                    self.logger.warning("The task %s of audio %s is failed, detail: %s", task_id, audio, metadata)
                    return {**self.default_result, "failed_reason": metadata.get("business_code", "FAILED_TASK_STATUS")}
                elif task_status in ["ACCEPTED", "PENDING", "RUNNING"]:
                    waiting = self.wait_exponential_jitter(attempts + 1)
                    self.logger.debug(
                        "Task: %s doesn't finished, audio: %s, status: %s, sleep %fs",
                        task_id,
                        audio,
                        task_status,
                        waiting,
                    )
                    await asyncio.sleep(waiting)
                else:
                    await self.stats.log_failed()
                    clean_polling_stats()

                    self.logger.warning(
                        "Unexpected task status: %s, task: %s, audio: %s, skip.", task_status, task_id, audio
                    )
                    return {**self.default_result, "failed_reason": "UNEXPECTED_TASK_STATUS"}

                attempts += 1
                cost = (datetime.now() - start).total_seconds()
                if cost > self.max_polling_seconds:
                    await self.stats.log_failed()
                    clean_polling_stats()
                    self.logger.warning(
                        "Elapsed max query time: %ss, audio: %s, task: %s.", self.max_polling_seconds, audio, task_id
                    )
                    return {**self.default_result, "failed_reason": "ELAPSED_POLLING_TIME"}

                if attempts > self.max_polling_num:
                    await self.stats.log_failed()
                    clean_polling_stats()

                    self.logger.warning(
                        "Elapsed max query number: %s, audio: %s, task: %s.", self.max_polling_num, audio, task_id
                    )
                    return {**self.default_result, "failed_reason": "ELAPSED_POLLING_NUM"}

                self.polling_stats[audio] = {"acc_cost": cost, "num": attempts}
                self.logger.info("The polling stats: %s", self.polling_stats)
            except Exception as e:
                await self.stats.log_failed()
                clean_polling_stats()

                self.logger.exception("Failed to query task status, audio: %s, task: %s.", audio, task_id)
                return {**self.default_result, "failed_reason": str(e)}

    def wait_exponential_jitter(self, attempt: int) -> float:
        jitter = random.uniform(0, self.waiting_jitter)
        try:
            exp = self.waiting_exp_base ** (attempt - 1)
            result = self.waiting_initial * exp + jitter
        except OverflowError:
            result = self.waiting_max_seconds

        return max(0, min(result, self.waiting_max_seconds))
