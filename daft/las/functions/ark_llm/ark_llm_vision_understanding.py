# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.ark_llm.llm_generate_utils import gen_media_data
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

    **输入输出规范：**
    - 输入格式：
        - 图片/视频数据：string类型，支持base64编码/url地址
        - （可选配置）用户提示词：string类型，当用户需要为每条数据指定不同提示词时，传入用户提示词。若不传入，则使用prompt字段配置的统一提示词）
    - 输出格式：
        - 默认模式：str类型生成结果
        - 诊断模式：设置环境变量 LAS_LLM_FINISH_REASON_CHECK=true，返回完整的生成结果和模型结果结束原因：
            - llm_result：str类型，生成结果
            - finish_reason：str类型，模型结果结束原因，取值范围：stop、length、content_filter

    **支持模型示例：**
    - 豆包多模态模型示例：
        - doubao-1.5-vision-pro-32k（版本250115）
        - doubao-1.5-vision-lite（版本250315）
        - doubao-1.5-vision-pro（版本250328）
    """  # noqa: D415

    def __init__(
        self,
        model: str,
        version: str,
        access_key: str | None = None,
        account_id: str | None = None,
        inference_type: str = DEFAULT_INFERENCE_TYPE,
        system_text: str | None = None,
        system_image_url: str | None = None,
        system_video_url: str | None = None,
        prompt: str | None = None,
        multimodal_type: str = "image",
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
        **kwargs: dict[str, Any],
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
            prompt: 用户提示词，
                用户提示词，用于指导模型的行为。配置该字段时，会和输入的文本拼接，以user角色方式输入给模型。同时，该字段也可以配置为{query}，此时，输入的文本会替换掉该字段.
            multimodal_type: 媒体内容类型
                指定处理的是图像还是视频，默认是 image。可选值:
                - image: 图片
                - video: 视频
                - text: 文本
            image_format: 图片编码格式
                仅在 multimodal_type=image 时生效，默认 jpeg。支持格式: JPEG, PNG, WEBP,GIF, BMP, TIFF等常见格式。详细格式请参考 https://www.volcengine.com/docs/82379/1362931#%E5%9B%BE%E7%89%87%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E
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
        self.system_text = system_text
        self.system_image_url = system_image_url
        self.system_video_url = system_video_url
        self.prompt = prompt
        self.image_url_detail = image_url_detail
        self.video_fps = video_fps

        self.source_type = source_type.lower() if source_type else "url"
        assert self.source_type in ["binary", "base64", "url"], "source_type must be binary, base64 or url"

        self.multimodal_type = multimodal_type.lower() if multimodal_type else "image"
        assert self.multimodal_type in ["image", "video", "text"], "multimodal_type must be image, video, text"

        self.image_format = image_format.lower() if image_format else "jpeg"
        self.video_format = video_format.lower() if video_format else "mp4"

        super().__init__(
            model=model,
            version=version,
            access_key=access_key,
            account_id=account_id,
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

    def transform(self, media_datas: pa.Array, user_prompts: pa.Array | None = None) -> pa.Array:
        """批量使用大模型进行视频理解.

        该方法使用预加载的大模型对输入的文本数组进行批量推理，生成对应的模型输出结果。

        Args:
            media_datas: 传入待处理的图片或视频数据。支持传入图片或视频的base64编码或url
            user_prompts: 传入用户提示词。当传入图片或视频数据时，若图片或视频数据使用的提示词不同时，可以通过该字段指定。若相同，则可以通过prompt参数指定。

        Returns:
            （默认情况下）当环境变量LAS_LLM_FINISH_REASON_CHECK=false时，返回字段类型为str。
            当环境变量LAS_LLM_FINISH_REASON_CHECK=true时，返回字段类型为struct，包含以下字段：
                - llm_result: 模型输出结果
                - finish_reason: 模型输出结束原因
        """
        message_generator = {
            "image": self._build_image_message,
            "video": self._build_video_message,
            "text": self._build_text_message,
        }[self.multimodal_type]

        media_list = media_datas.to_pylist()
        text_list = user_prompts.to_pylist() if user_prompts else [None] * len(media_datas)

        model_messages: list[list[dict[str, Any]]] = [
            message_generator(media_data=media, user_prompt=text) for media, text in zip(media_list, text_list)
        ]

        return super().process(model_messages)

    def _build_image_message(self, media_data: Any, user_prompt: str | None = None) -> list[dict[str, Any]]:
        """Build image message structure."""
        image_content = self._create_image_content(media_data)
        return self._assemble_message(image_content=image_content, user_prompt=user_prompt)

    def _build_video_message(self, media_data: Any, user_prompt: str | None = None) -> list[dict[str, Any]]:
        """Build video message structure."""
        video_content = self._create_video_content(media_data)
        return self._assemble_message(video_content=video_content, user_prompt=user_prompt)

    def _build_text_message(self, media_data: Any, user_prompt: str | None = None) -> list[dict[str, Any]]:
        text_content = {"type": "text", "text": media_data}
        return self._assemble_message(text_content=text_content, user_prompt=user_prompt)

    def _create_image_content(self, media_data: Any) -> dict[str, Any]:
        """Create image content structure."""
        media_url_or_data = gen_media_data("image", media_data, self.image_format, self.source_type)
        image_info: dict[str, Any] = {"url": media_url_or_data}
        if self.image_url_detail:
            image_info["detail"] = self.image_url_detail
        return {"type": "image_url", "image_url": image_info}

    def _create_video_content(self, media_data: Any) -> dict[str, Any]:
        """Create video content structure."""
        media_url_or_data = gen_media_data("video", media_data, self.video_format, self.source_type)
        video_info: dict[str, Any] = {"url": media_url_or_data}
        if self.video_fps is not None:
            video_info["fps"] = self.video_fps
        return {"type": "video_url", "video_url": video_info}

    def _assemble_message(
        self,
        *,
        text_content: dict[str, Any] | None = None,
        image_content: dict[str, Any] | None = None,
        video_content: dict[str, Any] | None = None,
        user_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        user_content = []
        if prompt_text := (user_prompt or self.prompt):
            user_content.append({"type": "text", "text": prompt_text})

        if text_content:
            user_content.append(text_content)

        if image_content:
            user_content.append(image_content)
        if video_content:
            user_content.append(video_content)

        messages = [{"role": "user", "content": user_content}]

        if system_content := self._build_system_message():
            messages.insert(0, {"role": "system", "content": system_content})

        return messages

    def _build_system_message(self) -> list[dict[str, Any]]:
        system_content = []
        if self.system_image_url:
            system_content.append(self._create_image_content(self.system_image_url))
        if self.system_video_url:
            system_content.append(self._create_video_content(self.system_video_url))
        if self.system_text:
            system_content.append({"type": "text", "text": self.system_text})
        return system_content
