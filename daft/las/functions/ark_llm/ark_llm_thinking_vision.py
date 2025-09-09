# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import json
import logging
import os
from typing import Any

from daft.dependencies import pa
from daft.las.functions.ark_llm.ark_llm_vision_understanding import ArkLLMVisionUnderstanding
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


DEFAULT_LAS_LLM_FINISH_REASON_CHECK = os.getenv("LAS_LLM_FINISH_REASON_CHECK", "false").lower() == "true"
DEFAULT_LAS_LLM_BOTS_REFERENCES = os.getenv("LAS_LLM_BOTS_REFERENCES", "false").lower() == "true"


class ArkLLMThinkingVision(ArkLLMVisionUnderstanding):
    """**多模态场景下提供大模型的深度思考能力**

    使用具备深度思考能力的模型进行图片、视频或文本进行分析理解，并返回文本输出.

    **核心功能：**
    - 深度思考机制：模型在回答问题前自动进行问题拆解和逻辑推理，生成思维链（reasoning_content）
    - 多模态场景支持：支持图片/视频理解任务，自动构建符合多模态模型规范的message结构
    - 输入简化机制：配置图片/视频的base64编码、URL等输入格式，便可以实现视觉理解功能

    **输入输出规范：**
    - 输入格式：
        - 图片/视频数据/文本数据：string类型，支持base64编码/url地址
        - （可选）用户提示词：string类型，当需要为每条数据指定不同提示词时传入，未传入时使用类初始化参数中的prompt
    - 输出格式：
        - 默认模式：struct类型包含 llm_result（生成结果）、reasoning_content（思维链内容）和 finish_reason（模型结果结束原因）
        - 诊断模式：设置环境变量 LAS_LLM_FINISH_REASON_CHECK=true，额外返回 finish_reason 字段：
            - finish_reason：模型结果结束原因，取值范围：stop（正常终止）、length（超出token限制）、content_filter（内容过滤）

    **模型能力增强：**
    - 思维链可视化：通过 reasoning_content 字段输出模型的推理过程
    - 结果可靠性控制：通过 finish_reason 字段识别异常终止情况
    - 多模态理解：支持图片/视频/文本的混合输入解析
    """  # noqa: D415

    _finish_reason_check = DEFAULT_LAS_LLM_FINISH_REASON_CHECK
    _bots_references = DEFAULT_LAS_LLM_BOTS_REFERENCES

    def __init__(
        self,
        model: str,
        version: str | None = None,
        thinking_type: str | None = None,
        multimodal_type: str = "image",
        **kwargs: Any,
    ) -> None:
        """提供基于火山方舟平台的大模型服务，使用具备深度思考能力的模型进行图片、视频或文本进行分析理解，并返回文本输出.

        深度思考能力的模型，可以提升最终答案的准确性。模型在回答问题前，会对问题进行分析和拆解，并基于对问题的拆解回答问题，回答会更加全面和深入。
        当您向模型提问时，方舟返回模型回答问题前的问题思考逻辑（思维链内容），基于此可观察模型推导过程并使用这部分信息。

        Args:
            model: 模型名称
                支持的模型有:豆包模型和DeepSeek模型。 示例 doubao-seed-1.6
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
                仅在 multimodal_type=image时生效，默认 jpeg。支持格式: JPEG, PNG, WEBP,GIF, BMP, TIFF等常见格式。详细格式请参考 https://www.volcengine.com/docs/82379/1362931#%E5%9B%BE%E7%89%87%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E
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
            thinking_type: 思考模式
                控制模型是否开启深度思考模式。不配置时，采用深度思考模式，可以手动关闭。可选值：
                - enabled：开启思考模式，模型一定先思考后回答。
                - disabled：关闭思考模式，模型直接回答问题，不会进行思考。
                - auto：自动思考模式，模型根据问题自主判断是否需要思考，简单题目直接回答。
            llm_config: 自定义LLM配置
                除了上述参数，其他参数会透传给模型。上述参数会覆盖llm_config中的值
            request_timeout: 超时时间
                单次请求的超时时间（秒）
            max_concurrency: 并发数
                每个进程的最大并发数
        """
        llm_config = kwargs.pop("llm_config", {})
        if thinking_type:
            thinking_type = thinking_type.lower()
            llm_config["thinking"] = {"type": thinking_type}

        super().__init__(
            model=model,
            version=version,
            llm_config=llm_config,
            multimodal_type=multimodal_type,
            **kwargs,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=model)

    def transform(self, media_datas: pa.Array, user_prompts: pa.Array | None = None) -> pa.Array:
        """批量使用大模型进行视频理解.

        该方法使用预加载的大模型对输入的文本数组进行批量推理，生成对应的模型输出结果。

        Args:
            media_datas: 传入待处理的图片或视频数据。支持传入图片或视频的base64编码或url
            user_prompts: 传入用户提示词。当传入图片或视频数据时，若图片或视频数据使用的提示词不同时，可以通过该字段指定。若相同，则可以通过prompt参数指定。

        Returns:
                当环境变量LAS_LLM_FINISH_REASON_CHECK=true时，返回字段类型为struct，包含以下字段：
                    - llm_result: 模型输出结果
                    - finish_reason: 模型输出结束原因
                    - reasoning_content: 模型输出思考内容
                当环境变量LAS_LLM_FINISH_REASON_CHECK=false时，返回字段类型为struct，包含以下字段：
                    - llm_result: 模型输出结果
                    - reasoning_content: 模型输出思考内容
        """
        return super().transform(media_datas, user_prompts)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        if ArkLLMThinkingVision._bots_references:
            return pa.struct(
                {
                    "llm_result": pa.string(),
                    "references": pa.string(),
                    "finish_reason": pa.string(),
                    "reasoning_content": pa.string(),
                }
            )
        if ArkLLMThinkingVision._finish_reason_check:
            return pa.struct(
                {"llm_result": pa.string(), "finish_reason": pa.string(), "reasoning_content": pa.string()}
            )

        return pa.struct({"llm_result": pa.string(), "reasoning_content": pa.string()})

    def _update_array_with_results(self, results: list[dict[str, Any] | None]) -> tuple[pa.Array, pa.Array]:
        # init output_data and finish_reason_data
        output_data = [None] * len(results)
        finish_reason_data: list[str | None] = [None] * len(results)
        reasoning_content_data = [None] * len(results)
        references_data: list[str | None] = [None] * len(results)

        for i, result in enumerate(results):
            if result is None:
                output_data[i] = None
                finish_reason_data[i] = "skip_empty_payload"
                reasoning_content_data[i] = None
                continue

            output_data[i] = result.get("choices", [{}])[0].get("message", {}).get("content")
            reasoning_content_data[i] = result.get("choices", [{}])[0].get("message", {}).get("reasoning_content")

            if ArkLLMThinkingVision._bots_references:
                references = result.get("references")
                references_data[i] = json.dumps(references, ensure_ascii=False) if references is not None else None
                reasoning_content_data[i] = result.get("choices", [{}])[0].get("message", {}).get("reasoning_content")
            elif ArkLLMThinkingVision._finish_reason_check:
                if "error" in result:
                    finish_reason_data[i] = result.get("error")
                else:
                    finish_reason_data[i] = result.get("choices", [{}])[0].get("finish_reason")

        output_array = pa.array(output_data, type=pa.string())
        reasoning_content_array = pa.array(reasoning_content_data, type=pa.string())

        if ArkLLMThinkingVision._bots_references:
            finish_reason_array = pa.array(finish_reason_data, type=pa.string())
            references_array = pa.array(references_data, type=pa.string())
            return pa.StructArray.from_arrays(
                [output_array, references_array, finish_reason_array, reasoning_content_array],
                ["llm_result", "references", "finish_reason", "reasoning_content"],
            )

        if ArkLLMThinkingVision._finish_reason_check:
            finish_reason_array = pa.array(finish_reason_data, type=pa.string())
            return pa.StructArray.from_arrays(
                [output_array, finish_reason_array, reasoning_content_array],
                ["llm_result", "finish_reason", "reasoning_content"],
            )

        return pa.StructArray.from_arrays([output_array, reasoning_content_array], ["llm_result", "reasoning_content"])
