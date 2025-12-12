from __future__ import annotations

import asyncio
import json
import logging
import os
from logging import Logger
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx

from daft.dependencies import pa
from daft.las.functions.operator import DEFAULT_LAS_OPERATOR_ENDPOINT, LasOperator, LasPollOperator
from daft.las.functions.types import AsyncOperatorStats, EventLooper
from daft.las.functions.utils.common_utils import tracking_usage


class LasAsrSubmitter(LasOperator):
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
        api_key: str | None = None,
        endpoint: str = DEFAULT_LAS_OPERATOR_ENDPOINT,
        version: str = "v1",
        uid: str | None = None,
        operator_id: str = "las_asr",
        operator_version: str = "v2",
        num_coroutines: int = 20,
        max_retries: int = 3,
        model_version: str = "400",
        enable_punc: bool | None = None,
        enable_ddc: bool | None = None,
        enable_speaker_info: bool | None = None,
        enable_itn: bool | None = None,
        enable_channel_split: bool | None = None,
        show_speech_rate: bool | None = None,
        show_volume: bool | None = None,
        enable_lid: bool | None = None,
        enable_emotion_detection: bool | None = None,
        enable_gender_detection: bool | None = None,
        enable_poi_fc: bool | None = None,
        enable_music_fc: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """创建一个 LasAsrSubmitter 实例，用于提交音频到LAS ASR 服务进行处理.

        Args:
            api_key: LAS 服务 API Key。若为空，将从环境变量 `LAS_API_KEY` 读取。
            endpoint: LAS 服务 API Endpoint。若为空，将读取环境变量 `LAS_SERVICE_ENDPOINT`。
            version: API 版本，默认 "v1"。
            uid: 用户唯一标识，默认 None。
            operator_id: LAS ASR服务版本ID，默认 "las_asr"。
            operator_version: LAS ASR服务版本，默认 "v1"。
            num_coroutines: 单实例并发提交的最大数量，默认 20。
            max_retries: 单次 API 请求最大重试次数，默认 3。
            model_version: 模型版本，传 "400" 使用 400 模型，默认 310。
            enable_punc: 是否开启标点补全。
            enable_ddc: 是否开启语义顺滑（DDC）。
            enable_speaker_info: 是否包含说话人信息。
            enable_itn: 是否开启文本规范化（ITN）。
            enable_channel_split: 是否开启通道分离。
            show_speech_rate: 是否返回语速。
            show_volume: 是否返回音量。
            enable_lid: 是否开启语种识别。
            enable_emotion_detection: 是否开启情绪检测。
            enable_gender_detection: 是否开启性别检测。
            enable_poi_fc: 是否开启 POI 领域推荐词。
            enable_music_fc: 是否开启音乐领域推荐词。
            **kwargs: 额外请求参数，包括：
                 `model_name`：默认从环境变量 `LAS_AUDIO_ASR_MODEL_NAME` 读取，缺省为 "bigmodel"），
                `show_utterances`：输出语音停顿、分句、分词信息。
                `vad_segment`：打开双声道识别时，通常需要使用vad分句，可同时打开此参数，默认False。
                `end_window_size`：强制判停时间范围300 - 5000ms，建议设置800ms或者1000ms，比较敏感的场景可以配置500ms或者更小。
                `corpus`: 领域推荐词，默认 None。·

        Returns:
            None

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
        self.logger = logging.getLogger(f"LasAsrSubmitter-{id(self)}")

        if uid is None:
            self.user_info = None
        else:
            self.user_info = {
                "user": {"uid": uid},
            }

        # construct request options
        self.request_options: dict[str, Any] = {
            "model_name": kwargs.pop("model_name", os.getenv("LAS_AUDIO_ASR_MODEL_NAME", "bigmodel")),
            "model_version": model_version,
            "enable_itn": enable_itn,
            "enable_punc": enable_punc,
            "enable_ddc": enable_ddc,
            "enable_speaker_info": enable_speaker_info,
            "enable_channel_split": enable_channel_split,
            "show_utterances": kwargs.pop("show_utterances", None),
            "show_speech_rate": show_speech_rate,
            "show_volume": show_volume,
            "enable_lid": enable_lid,
            "enable_emotion_detection": enable_emotion_detection,
            "enable_gender_detection": enable_gender_detection,
            "vad_segment": kwargs.pop("vad_segment", None),
            "end_window_size": kwargs.pop("end_window_size", None),
            "sensitive_words_filter": kwargs.pop(
                "sensitive_words_filter", os.getenv("LAS_AUDIO_ASR_SENSITIVE_WORDS_FILTER")
            ),
            "enable_poi_fc": enable_poi_fc,
            "enable_music_fc": enable_music_fc,
            "corpus": kwargs.pop("corpus", None),
        }
        self.request_options = {k: v for k, v in self.request_options.items() if v is not None}

        self.semaphore = asyncio.Semaphore(num_coroutines)
        self.stats = AsyncOperatorStats(self.logger)
        self.event_loop = EventLooper()

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="openspeech")

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()

    def transform(self, audio_urls: pa.Array, audio_metas: pa.Array | None = None) -> pa.Array:
        results = self.event_loop.run(
            self.run(audio_urls.to_pylist(), audio_metas.to_pylist() if audio_metas else [None] * len(audio_urls))
        )
        return pa.array(results, type=self.__return_column_type__())

    async def run(self, audio_urls: list[str], audio_metas: list[dict[str, Any] | None]) -> list[str]:
        await self.stats.log_accept(len(audio_urls))

        async def _submit(url: str, audio_meta: dict[str, Any] | None = None) -> str:
            async with self.semaphore:
                return await self.submit_req(url, audio_meta)

        # log the process before current batch
        await self.stats.log_process()
        result = await asyncio.gather(
            *[_submit(url, dict(meta) if meta else None) for url, meta in zip(audio_urls, audio_metas)]
        )
        # log the process after current batch
        await self.stats.log_process()

        return result

    async def submit_req(self, url: str, audio_meta: dict[str, Any] | None = None) -> str:
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
            payload = await self._generate_payload(audio_meta, url)
            resp = await self.submit(data=payload)
            return await self.parse_response(resp, url)
        except Exception:
            await self.stats.log_failed()
            self.logger.exception("Failed to sumit url: %s.", url)
            return ""

    async def parse_response(self, resp: httpx.Response, url: str) -> Any:
        meta = resp.json().get("metadata", {})
        self.logger.debug("The metadata of response for audio: %s is %s", url, meta)

        if resp.status_code != 200:
            await self.stats.log_failed()
            attempt_num = resp.headers.get("attempt_num", 0)
            req_id = meta.get("request_id", "")
            self.logger.warning(
                "Failed to submit url: %s, http status: %s, attempt: %s, req_id: %s, detail: %s",
                url,
                resp.status_code,
                attempt_num,
                req_id,
                resp.text,
            )
            return ""

        await self.stats.log_succeed()
        return meta.get("task_id", "")

    async def _generate_payload(self, audio_meta: dict[str, Any] | None, url: str) -> dict[str, Any]:
        request_payload = dict(self.request_options)
        if audio_meta:
            corpus = audio_meta.get("corpus")
            if corpus:
                request_payload["corpus"] = corpus

        audio_payload = self._generate_audio_payload(audio_meta, url)
        data_payload = {
            "audio": audio_payload,
            "request": request_payload,
        }

        if self.user_info:
            data_payload["user"] = self.user_info

        self.logger.debug("Submitting LAS ASR request for audio: %s, data payload: %s", url, data_payload)
        return data_payload

    def _generate_audio_payload(self, audio_meta: dict[str, Any] | None, url: str) -> dict[str, Any]:
        self.logger.debug("Submitting LAS ASR audio %s with meta %s", url, audio_meta)

        filename = os.path.basename(urlparse(url).path)
        ext = filename.split(".")[-1].lower() if (filename and "." in filename) else ""
        guessed_format = ext if ext in {"wav", "mp3", "aac", "flac", "ogg"} else None

        audio_payload = {
            "url": url,
        }

        if audio_meta:
            audio_optional = {
                "language": audio_meta.get("language"),
                "codec": audio_meta.get("codec"),
                "rate": audio_meta.get("rate"),
                "bits": audio_meta.get("bits"),
                "channel": audio_meta.get("channel"),
                "format": audio_meta.get("format"),
            }
            audio_payload.update({k: v for k, v in audio_optional.items() if v is not None})

        if not audio_payload.get("format") and guessed_format:
            audio_payload["format"] = guessed_format

        return audio_payload


class LasAsrPoller(LasPollOperator):
    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = DEFAULT_LAS_OPERATOR_ENDPOINT,
        version: str = "v1",
        operator_id: str = "las_asr",
        operator_version: str = "v2",
        num_coroutines: int = 20,
        max_retries: int = 5,
        max_polling_seconds: int = 7200,
        max_polling_num: int = 240,
        waiting_initial: float = 1,
        waiting_exp_base: float = 2,
        waiting_jitter: float = 1,
        waiting_max_seconds: float = 30,
        enable_speaker_info: bool = False,
        enable_channel_split: bool = False,
        **kwargs: Any,
    ):
        """创建一个 LasAsrPoller 实例，并发轮询任务状态与结果（Poll）.

        - 异步流程：携带 `task_id` 调用（POST /api/v1/poll）直至 `COMPLETED`/`FAILED`/`TIMEOUT`
        - 输出结构保持向后兼容：`{"asr_result_raw": str(JSON), "asr_result_text": text, "failed_reason": msg}`

        Args:
            api_key: LAS 服务 API Key。若为空，将从环境变量 `LAS_API_KEY` 读取。
            endpoint: LAS 服务 API Endpoint。若为空，将读取环境变量 `LAS_SERVICE_ENDPOINT`。
            version: API 版本，默认 "v1"。
            operator_id: 算子 ID，默认 "las_asr"。
            operator_version: 算子版本，默认 "v2"。
            num_coroutines: 并发轮询的最大数量，默认 20。
            max_retries: 单次轮询请求最大重试次数，默认 5。
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
            num_coroutines=num_coroutines,
            max_polling_seconds=max_polling_seconds,
            max_polling_num=max_polling_num,
            waiting_initial=waiting_initial,
            waiting_exp_base=waiting_exp_base,
            waiting_jitter=waiting_jitter,
            waiting_max_seconds=waiting_max_seconds,
            **kwargs,
        )

        self.enable_speaker_info = enable_speaker_info
        self.enable_channel_split = enable_channel_split

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("asr_result_raw", pa.string()),
                pa.field("asr_result_text", pa.string()),
                pa.field("asr_result_simple", pa.string()),
                pa.field("failed_reason", pa.string()),
            ]
        )

    def transform(self, audios: pa.Array, tasks: pa.Array) -> pa.Array:
        results = self.batch_poll(tasks.to_pylist(), audios.to_pylist())
        return pa.array(results, type=self.__return_column_type__())

    def get_logger(self) -> Logger:
        return logging.getLogger(f"LasAsrPoller-{id(self)}")

    def get_default_result(self) -> dict[str, str]:
        return {"asr_result_raw": "", "asr_result_simple": "", "asr_result_text": ""}

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        result_data = data.get("result", {})
        utts = result_data.get("utterances", [])
        formatted_result = self.format_asr_text(utts)

        try:
            raw_str = json.dumps(result_data, ensure_ascii=False)
        except Exception:
            raw_str = str(result_data)

        return {
            "asr_result_raw": raw_str,
            "asr_result_simple": formatted_result,
            "asr_result_text": result_data.get("text", ""),
        }

    def ms_to_hms(self, ms: int) -> str:
        """将毫秒转换为 hh:mm:ss 格式."""
        ms //= 1000
        h, ms = divmod(ms, 3600)
        m, s = divmod(ms, 60)
        return f"{h}:{m:02d}:{s:02d}"

    def format_asr_text(self, utts: list[Any]) -> str:
        lines = []

        if self.enable_speaker_info:
            for u in utts:
                lines.append(
                    f"说话人 {u['additions']['speaker']} "
                    f"{self.ms_to_hms(u['start_time'])} {self.ms_to_hms(u['end_time'])}\n{u['text']}\n"
                )
        elif self.enable_channel_split:
            for u in utts:
                lines.append(
                    f"通道 {u['additions']['channel_id']} "
                    f"{self.ms_to_hms(u['start_time'])} {self.ms_to_hms(u['end_time'])}\n{u['text']}\n"
                )
        else:
            for u in utts:
                lines.append(f"{self.ms_to_hms(u['start_time'])} {self.ms_to_hms(u['end_time'])} {u['text']}")

        return "\n".join(lines).strip()
