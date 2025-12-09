# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.ark_llm.llm_generate_utils import gen_media_data
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.infra.las_ark import (
    DEFAULT_INFERENCE_TYPE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from daft.dependencies import pa


class ArkLLMVisionUnderstanding(ArkLLMGenerate):
    """**大模型多模态视频理解处理器**

    **核心功能：**
    - 多模态场景支持：支持图片/视频理解任务，自动构建符合多模态模型规范的message结构
    - 输入简化机制：配置图片/视频的base64编码、URL等输入格式，便可以实现视觉理解功能
    - 多种数据源支持：支持本地文件路径、HTTP/HTTPS URL、TOS/S3对象存储等多种数据源
    - 灵活的输入组合：支持单独使用文本、图片、视频，或任意组合使用

    **输入输出规范：**
    - 输入格式：
        - 图片：string类型/列表类型，支持base64编码、binary数据格式、HTTP/HTTPS URL、TOS地址
        - 视频：string类型/列表类型，支持base64编码、binary数据格式、HTTP/HTTPS URL、TOS地址
        - 文本：string类型/列表类型，用户输入的文本
    - 输出格式：
        - 默认模式：str类型生成结果
        - 诊断模式：设置环境变量 LAS_LLM_FINISH_REASON_CHECK=true，返回完整的生成结果和模型结果结束原因：
            - llm_result：str类型，生成结果
            - finish_reason：str类型，模型结果结束原因，取值范围：stop、length、content_filter
    """  # noqa: D415

    def __init__(
        self,
        model: str,
        version: str | None = None,
        inference_type: str = DEFAULT_INFERENCE_TYPE,
        system_text: str | None = None,
        system_image_url: str | None = None,
        system_video_url: str | None = None,
        image_format: str = "jpeg",
        image_url_detail: str | None = None,
        video_format: str = "mp4",
        video_fps: float | None = None,
        source_type: str = "url",
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        temperature: float = 1,
        top_p: float = 0.7,
        logit_bias: dict[str, Any] | None = None,
        tools: list[dict[Any, Any]] | None = None,
        llm_config: dict[str, Any] | None = None,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        **kwargs: Any,
    ) -> None:
        """提供基于火山方舟平台的大模型服务，进行图片或视频理解，返回文本输出.

        Args:
            model: 模型名称
                支持的模型有:豆包模型和DeepSeek模型。 示例 doubao-1.5-lite-32k
            version: 模型版本
                输入模型对应的版本信息。示例 250115
            inference_type: 推理类型，支持在线推理和批量推理。默认值为batch，即采用批量推理
                - online： 采用方舟平台提供的在线推理模块进行推理
                - batch：采用方舟平台提供的批量推理模块进行推理
            system_text: 系统提示内容
                系统提示内容，以system角色作为模型的输入
            system_image_url: 系统图片 URL
                图文混排场景下，输入系统图片 URL，用于指导模型的行为
            system_video_url: 系统视频 URL
                图文混排场景下，输入系统视频 URL，用于指导模型的行为
            image_format: 图片编码格式
                默认 jpeg。支持格式: JPEG, PNG, WEBP,GIF, BMP, TIFF等常见格式。详细格式请参考 https://www.volcengine.com/docs/82379/1362931#%E5%9B%BE%E7%89%87%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E
            image_url_detail: 图片质量
                支持手动设置图片的质量，取值范围high、low、auto。
                - high：高细节模式，适用于需要理解图像细节信息的场景，如对图像的多个局部信息/特征提取、复杂/丰富细节的图像理解等场景，理解更全面。
                - low：低细节模式，适用于简单的图像分类/识别、整体内容理解/描述等场景，理解更快速。
                - auto：默认模式，不同模型选择的模式略有不同，具体请参见https://www.volcengine.com/docs/82379/1362931#bf4d9224。
            video_format: 视频编码格式
                配置视频格式，默认是mp4。支持的视频格式：MP4、AVI、MOV。单视频文件需在 50MB 以内。
            video_fps: 视频帧率
                取值范围：[0.2, 5]。默认值 1
                每秒钟从视频中抽取指定数量的图像。取值越高，对于视频中画面变化理解越精细；取值越低，对于视频中画面变化感知减弱，但是使用的 token 花费少，速度也更快。
                请参考https://www.volcengine.com/docs/82379/1362931#%E7%94%A8%E9%87%8F%E8%AF%B4%E6%98%8E
            source_type: 数据来源类型
                指定媒体数据的来源格式，默认 url。可选值:
                - binary: 原始二进制数据
                - base64: Base64编码数据
                - url: 网络资源地址（支持 http/https/tos）
            max_tokens:
                模型回复最大长度（单位 token），输入输出总长度受模型上下文限制
            max_completion_tokens:
               模型生成的 token 数量的上限，包含思维链内容（reasoning_content）与回答内容（content），不包含传入的信息（messages）。
               超出后，停止模型输出思维链内容及模型回答，并返回finish_reason字段为length。
            stop:
                模型遇到 stop 字段所指定的字符串时将停止继续生成，这个词语本身不会输出。
                最多支持 4 个字符串。例如 ["你好", "天气"]
            frequency_penalty: 频率惩罚系数
                频率惩罚系数。如果值为正，会根据新 token 在文本中的出现频率对其进行惩罚，从而降低模型逐字重复的可能性。
                取值范围 [-2.0, 2.0]，默认0
            presence_penalty: 存在惩罚系数
                存在惩罚系数。如果值为正，会根据新 token 到目前为止是否出现在文本中对其进行惩罚，从而增加模型谈论新主题的可能性。
                取值范围为 [-2.0, 2.0]。默认值 0
            temperature:  采样温度
                采样温度。控制了生成文本时对每个候选词的概率分布进行平滑的程度。
                - 当取值为 0 时模型仅考虑对数概率最大的一个 token。
                - 较高的值（如 0.8）会使输出更加随机，而较低的值（如 0.2）会使输出更加集中确定。
                通常建议仅调整 temperature 或 top_p 其中之一，不建议两者都修改。取值范围为 [0, 2]。默认值 1
            top_p: 核采样概率阈值
                核采样概率阈值。模型会考虑概率质量在 top_p 内的 token 结果。
                当取值为 0 时模型仅考虑对数概率最大的一个 token。
                0.1 意味着只考虑概率质量最高的前 10% 的 token，取值越大生成的随机性越高，取值越低
                生成的确定性越高。通常建议仅调整 temperature 或 top_p 其中之一，不建议两者都修改。
                默认值 0.7
            logit_bias:
                调整指定 token 在模型输出内容中出现的概率，使模型生成的内容更加符合特定的偏好。
                logit_bias 字段接受一个 map 值，其中每个键为词表中的 token ID（使用 tokenization 接口获取），每个值为该 token 的偏差值，取值范围为 [-100, 100]。
                -1 会减少选择的可能性，1 会增加选择的可能性；-100 会完全禁止选择该 token，100 会导致仅可选择该 token。
                该参数的实际效果可能因模型而异。
            tools:
                待调用工具的列表，模型返回信息中可包含。当您需要让模型返回待调用工具时，需要配置该结构体。
            llm_config: 自定义LLM配置
                除了上述参数，其他参数会透传给模型。上述参数会覆盖llm_config中的值
            request_timeout: 超时时间
                单次请求的超时时间（秒）
            max_concurrency: 并发数
                每个进程的最大并发数
        """
        super().__init__(
            model=model,
            version=version,
            inference_type=inference_type,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            logit_bias=logit_bias,
            tools=tools,
            llm_config=llm_config,
            request_timeout=request_timeout,
            max_concurrency=max_concurrency,
            **kwargs,
        )

        self.system_text = system_text
        self.system_image_url = system_image_url
        self.system_video_url = system_video_url
        self.image_url_detail = image_url_detail
        self.video_fps = video_fps

        self.source_type = source_type.lower() if source_type else "url"
        assert self.source_type in ["binary", "base64", "url"], "source_type must be binary, base64 or url"

        self.image_format = image_format.lower() if image_format else "jpeg"
        self.video_format = video_format.lower() if video_format else "mp4"

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=model)

    def _flatten_if_list(self, item: Any) -> list[Any]:
        """Flatten nested lists or return the item as a single-item list."""
        if item is None:
            return []
        if isinstance(item, list):
            result = []
            for sub_item in item:
                result.extend(self._flatten_if_list(sub_item))
            return result
        return [item]

    def transform(
        self,
        images: pa.Array | str | None = None,
        videos: pa.Array | str | None = None,
        texts: pa.Array | str | None = None,
    ) -> pa.Array:
        """批量使用大模型进行视频理解.

        该方法使用火山方舟平台上的大模型对输入的图像、视频和文本数据进行批量推理，生成对应的模型输出结果。
        支持单独或组合使用图像、视频和文本输入，能够处理多种数据源格式（URL、Base64、二进制）。

        Args:
            images: 传入待处理的图片数据。支持传入图片的base64编码或url。支持传入单张图片，也支持以list方式传入多张图片。
                （但是不允许输入的图片中既包括单张图片的字符串类型，也包含list类型。）
                根据source_type参数的不同，图片数据会被相应处理：
                - url模式：支持http/https/tos/s3等协议的URL，其中tos/s3会生成预签名URL
                - base64模式：直接使用Base64编码数据
                - binary模式：将二进制数据转换为Base64编码

            videos: 传入待处理的视频数据。支持传入视频的base64编码或url。支持传入单条视频，也支持以list方式传入多条视频。
                （但是不允许输入的视频中既包括单条视频的字符串类型，也包含list类型。）
                根据source_type参数的不同，视频数据会被相应处理：
                - url模式：支持http/https/tos/s3等协议的URL，其中tos/s3会生成预签名URL
                - base64模式：直接使用Base64编码数据
                - binary模式：将二进制数据转换为Base64编码

            texts: 传入用户提示词。可以传入单条提示词，也可以以list方式传入多条提示词。
                （但是不允许输入的提示词中既包括单条提示词的字符串类型，也包含list类型。）

        Returns:
            （默认情况下）当环境变量LAS_LLM_FINISH_REASON_CHECK=false时，返回字段类型为str。
            当环境变量LAS_LLM_FINISH_REASON_CHECK=true时，返回字段类型为struct，包含以下字段：
                - llm_result: 模型输出结果
                - finish_reason: 模型输出结束原因
        """
        assert (
            images is not None or videos is not None or texts is not None
        ), "At least one of images, videos or texts must be provided."

        def normalize_input(input_data: pa.Array | None) -> tuple[list[Any] | None, int]:
            if input_data is None:
                return None, 0

            if hasattr(input_data, "to_pylist"):
                pylist = input_data.to_pylist()
                return pylist, len(pylist)
            elif isinstance(input_data, str) or not isinstance(input_data, (list, tuple)):
                return [input_data], 1
            else:
                pylist = list(input_data)
                return pylist, len(pylist)

        # Normalize inputs and get their lengths
        normalized_images, image_length = normalize_input(images)
        normalized_videos, video_length = normalize_input(videos)
        normalized_texts, text_length = normalize_input(texts)

        # Calculate the maximum length among all inputs
        data_len = max(image_length, video_length, text_length)

        messages_list = self._prepare_model_messages(normalized_images, normalized_videos, normalized_texts, data_len)
        return super().process(messages_list)

    def _prepare_model_messages(
        self, images: list[Any] | None, videos: list[Any] | None, texts: list[Any] | None, data_len: int = 0
    ) -> pa.Array:
        def prepare_list(input_list: list[Any] | None, length: int) -> list[Any]:
            if input_list is None:
                return [None] * length

            input_len = len(input_list)
            if input_len == 1 and length > 1:
                return input_list * length
            elif input_len < length:
                return input_list + [None] * (length - input_len)
            else:
                return input_list[:length]

        # Prepare lists with consistent length
        image_list = prepare_list(images, data_len)
        video_list = prepare_list(videos, data_len)
        text_list = prepare_list(texts, data_len)

        model_messages: list[list[dict[str, Any]]] = []
        for i in range(data_len):
            try:
                flat_images = self._flatten_if_list(image_list[i])
                flat_videos = self._flatten_if_list(video_list[i])
                flat_texts = self._flatten_if_list(text_list[i])
                user_content = []

                for text in flat_texts:
                    if text is not None:
                        user_content.append({"type": "text", "text": text})

                for image in flat_images:
                    image_content = self._create_image_content(image)
                    if image_content:
                        user_content.append(image_content)

                for video in flat_videos:
                    video_content = self._create_video_content(video)
                    if video_content:
                        user_content.append(video_content)

                messages = []
                if user_content:
                    messages.append({"role": "user", "content": user_content})

                if system_content := self._build_system_message():
                    messages.insert(0, {"role": "system", "content": system_content})

                model_messages.append(messages)
            except Exception as e:
                logger.error("Error preparing model messages for index %d: %s", i, str(e))
                model_messages.append([])
        return model_messages

    def _create_image_content(self, media_data: Any) -> dict[str, Any] | None:
        """Create image content structure."""
        if media_data is None:
            return None

        media_url_or_data = gen_media_data("image", media_data, self.image_format, self.source_type)
        image_info: dict[str, Any] = {"url": media_url_or_data}
        if self.image_url_detail:
            image_info["detail"] = self.image_url_detail
        return {"type": "image_url", "image_url": image_info}

    def _create_video_content(self, media_data: Any) -> dict[str, Any] | None:
        """Create video content structure."""
        if media_data is None:
            return None

        media_url_or_data = gen_media_data("video", media_data, self.video_format, self.source_type)
        video_info: dict[str, Any] = {"url": media_url_or_data}
        if self.video_fps is not None:
            video_info["fps"] = self.video_fps
        return {"type": "video_url", "video_url": video_info}

    def _build_system_message(self) -> list[dict[str, Any]]:
        system_content = []
        if image_content := self._create_image_content(self.system_image_url):
            system_content.append(image_content)
        if video_content := self._create_video_content(self.system_video_url):
            system_content.append(video_content)
        if self.system_text:
            system_content.append({"type": "text", "text": self.system_text})
        return system_content
