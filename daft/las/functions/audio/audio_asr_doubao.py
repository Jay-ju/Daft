# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx

from daft.dependencies import pa
from daft.las.functions.types import Operator

logger = logging.getLogger(__name__)


class AudioAsrDoubao(Operator):
    """**语音识别模块 - 基于豆包语音大模型的录音转写解决方案**

    **核心功能**
    - 接入火山引擎大模型（`volc.bigasr.auc`）的非流式录音识别接口
    - 支持自动断句、数字规整、说话人或通道分离（可选）
    - 并发处理多个音频文件，提供结构化 JSON 与可读文本两种输出
    - 适合转写最长 5 小时的录音文件，支持标点补全、智能断句、说话人分离等高级功能。

    **使用场景**
    - 智能会议纪要生成（推荐开启说话人识别）
    - 课后教学音频分析、学情回顾
    - 语音客服/外呼质检分析
    - 媒体节目、采访播客的离线字幕生成

    **企业接入说明**
    当前该能力仅对通过企业认证的用户开放，如需测试或正式接入，请先完成火山引擎企业认证流程。
    """  # noqa: D415

    def __init__(
        self,
        appid: str,
        token: str,
        uid: str,
        enable_punc: bool = True,
        enable_ddc: bool = True,
        enable_speaker_info: bool = True,
        enable_itn: bool = True,
        enable_channel_split: bool = False,
        poll_interval: int = 10,
        num_coroutines: int = 1,
        **kwargs: Any,
    ) -> None:
        """初始化 AudioAsrDoubao 类的实例

        Args:
            appid: 使用火山引擎控制台获取的 AppID，用于认证调用身份
            token: 使用火山引擎控制台获取的 Access Token，用于接口鉴权
            enable_punc: 是否启用自动断句与标点
            enable_ddc: 是否启用语义顺滑，输出更流畅的语音转写结果
            enable_speaker_info: 是否开启说话人分离，10人以内效果较好。
            enable_itn: 是否启用文本规范化，将 ASR 模型的原始语音输出转换为书面形式，以提高文本的可读性。
            enable_channel_split: 是否根据通道（channel_id）进行音频分轨处理
            poll_interval: 轮询查询识别结果的时间间隔
            num_coroutines: 并发处理音频的最大数量
            uid: 用户唯一标识
            **kwargs: 传递给父类 Operator 的其他关键字参数
        """  # noqa: D415
        super().__init__(**kwargs)
        self.appid = appid
        self.token = token
        self.enable_punc = enable_punc
        self.enable_ddc = enable_ddc
        self.enable_speaker_info = enable_speaker_info
        self.enable_itn = enable_itn
        self.enable_channel_split = enable_channel_split
        self.poll_interval = poll_interval
        self.num_coroutines = num_coroutines
        self.uid = uid
        self.submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
        self.query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

        self.client = httpx.AsyncClient()

    def generate_uid(self) -> str:
        return str(uuid.uuid4())

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

    async def process(self, audio_url: str) -> dict[str, str | None]:
        if not audio_url or not audio_url.strip():
            return {"asr_result_raw": None, "asr_result_simple": None, "asr_result_text": None}

        try:
            task_id = str(uuid.uuid4())

            headers = {
                "X-Api-App-Key": self.appid,
                "X-Api-Access-Key": self.token,
                "X-Api-Resource-Id": "volc.bigasr.auc",
                "X-Api-Request-Id": task_id,
                "X-Api-Sequence": "-1",
            }

            request = {
                "user": {"uid": self.uid},
                "audio": {"url": audio_url},
                "request": {
                    "model_name": "bigmodel",
                    "enable_punc": self.enable_punc,
                    "enable_ddc": self.enable_ddc,
                    "enable_speaker_info": self.enable_speaker_info,
                    "enable_itn": self.enable_itn,
                },
            }

            submit_resp = await self.client.post(self.submit_url, headers=headers, json=request, timeout=15)

            if submit_resp.headers.get("X-Api-Status-Code") != "20000000":
                logger.info(
                    "submit response for file %s: %s",
                    audio_url,
                    submit_resp.headers.get("X-Api-Message"),
                )
                return {"asr_result_raw": None, "asr_result_simple": None, "asr_result_text": None}

            logger.info(
                "submit response for file %s: %s",
                audio_url,
                submit_resp.headers.get("X-Api-Message"),
            )
        except Exception as e:
            logger.error("Submit error: %s. audio_url: %s", e, audio_url)
            return {"asr_result_raw": None, "asr_result_simple": None, "asr_result_text": None}

        try:
            while True:
                query_headers = {
                    "X-Api-App-Key": self.appid,
                    "X-Api-Access-Key": self.token,
                    "X-Api-Resource-Id": "volc.bigasr.auc",
                    "X-Api-Request-Id": task_id,
                }

                resp = await self.client.post(self.query_url, headers=query_headers, json={}, timeout=15)
                code = resp.headers.get("X-Api-Status-Code")

                logger.info(
                    "exec status response for file %s: %s",
                    audio_url,
                    resp.headers.get("X-Api-Message"),
                )

                if code == "20000000":
                    json_result = resp.json()
                    utts = json_result.get("result", {}).get("utterances", [])
                    formatted = self.format_asr_text(utts)
                    return {
                        "asr_result_raw": resp.text,
                        "asr_result_simple": formatted,
                        "asr_result_text": json_result.get("result", {}).get("text", ""),
                    }

                if code not in {"20000001", "20000002"}:
                    logger.error("Query failed. taskid: %s, status code: %s", task_id, code)
                    return {"asr_result_raw": None, "asr_result_simple": None, "asr_result_text": None}

                await asyncio.sleep(self.poll_interval)

        except Exception as e:
            logger.error("Query error: %s", e)
            return {"asr_result_raw": None, "asr_result_simple": None, "asr_result_text": None}

    async def async_run(self, audios: list[str]) -> list[dict[str, str | None]]:
        semaphore = asyncio.Semaphore(self.num_coroutines)

        async def bounded_process(v: str) -> dict[str, str | None]:
            async with semaphore:
                return await self.process(v)

        results = await asyncio.gather(*[bounded_process(v) for v in audios])
        return results

    def transform(self, audios: pa.Array) -> pa.Array:
        """批量处理音频链接，生成语音识别结构化结果。

        该方法使用豆包语音大模型语音识别能力，异步处理输入的音频链接列表，
        返回每段音频对应的识别结果，包括完整 JSON 和格式化文本两部分。

        Args:
            audios: 包含音频 URL 的数组，每个元素应为一个可访问的音频文件地址

        Returns:
            一个结构化结果数组，其中每个元素包含以下字段：
                - asr_result_raw (str): 完整的识别结果 JSON 字符串，包含时间戳、说话人等结构化信息
                - asr_result_simple (str): 提取后的转写文本，按说话人或时间段分段，适合直接阅读或展示
                - asr_result_text (str): 提取后的转写文本，仅包含转写内容
        """  # noqa: D415
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_run(audios.to_pylist()))
        return pa.array(results, type=AudioAsrDoubao.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("asr_result_raw", pa.string()),
                pa.field("asr_result_simple", pa.string()),
                pa.field("asr_result_text", pa.string()),
            ]
        )
