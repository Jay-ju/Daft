# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Any

import httpx

from daft.dependencies import pa
from daft.las.functions.types import Operator

logger = logging.getLogger(__name__)


class AudioTtsDoubao(Operator):
    """**语音合成模块 - 基于豆包语音大模型的文本转音频解决方案**

    **核心功能**
    - 接入火山引擎大模型（`volc.tts`）的语音合成接口
    - 支持多种参数配置，如音色、情绪、编码格式、语速、采样率等
    - 并发处理多个文本输入，输出 Base64 编码音频及原始响应
    - 适合用于语音播报、虚拟人声音生成、听力内容制作等场景

    **使用场景**
    - 智能客服、语音助理内容生成
    - 多语言文本播报系统
    - 内容平台、短视频配音
    - 虚拟人语音驱动

    **企业接入说明**
    当前该能力仅对通过企业认证的用户开放，如需测试或正式接入，请先完成火山引擎企业认证流程。
    """  # noqa: D415

    def __init__(
        self,
        appid: str,
        token: str,
        uid: str,
        cluster: str = "volcano_tts",
        voice_type: str = "zh_female_wanqudashu_moon_bigtts",
        encoding: str = "mp3",
        speed_ratio: float = 1.0,
        rate: int = 24000,
        bitrate: int = 160,
        enable_emotion: bool = False,
        emotion: str = "happy",
        emotion_scale: int = 4,
        extra_audio_params: dict[str, Any] | None = None,
        extra_request_params: dict[str, Any] | None = None,
        timeout: int = 60,
        num_coroutines: int = 1,
        **kwargs: Any,
    ) -> None:
        """初始化 AudioTtsDoubao 类的实例

        Args:
        appid: 火山引擎控制台获取的 AppID
        token: 火山引擎控制台获取的 Access Token
        cluster: 接入服务所使用的服务集群名
        uid: 用户标识，用于接口调用追踪
        voice_type: 音色类型，例如 "zh_female_wanqudashu_moon_bigtts"
        encoding: 音频编码格式（如 "mp3", "pcm"）
        speed_ratio: 语速调整比例（默认 1.0）
        rate: 采样率
        bitrate: 音频比特率（单位 kbps）
        enable_emotion: 是否启用情绪音色
        emotion: 情绪类型，如 "happy", "angry"
        emotion_scale: 情绪强度，范围 1~5
        extra_audio_params: 其他音频参数（可选，传入字典进行补充）
        extra_request_params: 其他请求参数（可选，传入字典进行补充）
        timeout: 单个请求的超时时间（秒）
        num_coroutines: 并发请求数量控制
        **kwargs: 传递给父类 Operator 的其他参数
        """  # noqa: D415
        super().__init__(**kwargs)
        self.appid = appid
        self.token = token
        self.cluster = cluster
        self.uid = uid

        self.voice_type = voice_type
        self.encoding = encoding
        self.speed_ratio = speed_ratio
        self.rate = rate
        self.bitrate = bitrate
        self.enable_emotion = enable_emotion
        self.emotion = emotion
        self.emotion_scale = emotion_scale
        self.extra_audio_params = extra_audio_params or {}
        self.extra_request_params = extra_request_params or {}

        self.timeout = timeout
        self.num_coroutines = num_coroutines

        self.url = "https://openspeech.bytedance.com/api/v1/tts"
        self.headers = {
            "Authorization": f"Bearer; {self.token}",
            "Content-Type": "application/json",
        }

        self.client = httpx.AsyncClient()

    def _build_audio_params(self) -> dict[str, Any]:
        audio = {
            "voice_type": self.voice_type,
            "encoding": self.encoding,
            "speed_ratio": self.speed_ratio,
            "rate": self.rate,
            "bitrate": self.bitrate,
        }

        if self.enable_emotion and self.emotion:
            audio["enable_emotion"] = True
            audio["emotion"] = self.emotion
            audio["emotion_scale"] = self.emotion_scale

        return {**audio, **self.extra_audio_params}

    async def process(self, text: str) -> bytes | None:
        if not text or not text.strip():
            return None

        try:
            task_id = str(uuid.uuid4())
            audio_params = self._build_audio_params()
            payload = {
                "app": {"appid": self.appid, "token": self.token, "cluster": self.cluster},
                "user": {"uid": self.uid},
                "audio": audio_params,
                "request": {
                    "reqid": task_id,
                    "text": text,
                    "operation": "query",
                    **self.extra_request_params,
                },
            }

            resp = await self.client.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)

            if resp.status_code != 200:
                logger.error("Request failed: %s - %s", resp.status_code, resp.text)
                return None

            resp_json = resp.json()
            audio_base64 = resp_json.get("data", "")
            if not audio_base64:
                return None

            return base64.b64decode(audio_base64)

        except Exception as e:
            logger.warning("TTS process error: %s", e)
            return None

    async def async_run(self, texts: list[str]) -> list[bytes | None]:
        semaphore = asyncio.Semaphore(self.num_coroutines)

        async def bounded_process(t: str) -> bytes | None:
            async with semaphore:
                return await self.process(t)

        results = await asyncio.gather(*[bounded_process(t) for t in texts])
        return results

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本输入，生成语音合成结果

        Args:
            texts: 输入文本的 PyArrow 数组，每个元素为字符串类型

        Returns:
            PyArrow 字符串数组，文本生成的音频结果；
            若识别失败或输入为空字符串，则对应元素为 None。
        """  # noqa: D415
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_run(texts.to_pylist()))
        return pa.array(results, type=AudioTtsDoubao.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.binary()
