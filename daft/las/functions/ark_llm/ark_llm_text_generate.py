# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daft.dependencies import pa
from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.ark_llm.llm_generate_utils import gen_text_message
from daft.las.functions.utils.common_utils import log_op_call
from daft.las.infra.las_ark import (
    DEFAULT_INFERENCE_TYPE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class ArkLLMTextGenerate(ArkLLMGenerate):
    """**大模型文本生成专用处理器（豆包/DeepSeek）**

    **核心功能：**
    - 纯文本场景优化：根据用户输入文本数据，自动构建符合模型规范的message结构
    - 输入简化机制：原生支持str类型输入，自动封装为{role: user, content: text}格式
    - 多任务支持：翻译/总结/问答等NLP场景开箱即用
    - 双提示词系统：
        - system_content：系统级行为指导（如翻译风格控制）
        - prompt：用户级指令模板（支持{query}占位符替换）

    **输入输出规范：**
    - 输入格式：纯文本数据
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
        system_content: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """针对纯文本的数据，调用方舟模型进行文本进行理解和回复。示例文本翻译、内容总结等场景，传入文本信息，通过大模型对这些文本信息作相应理解和回复.

        输入纯文本数据，将其按照方舟模型的输入格式进行组装message信息，格式为{role: user, content: <query语句>}。您只需要传入<query语句>即可.

        Args:
            model: 模型名称
                支持的模型有:豆包模型和DeepSeek模型。 示例 doubao-1.5-lite-32k
            version: 模型版本
                输入模型对应的版本信息。示例 250115
            inference_type: 推理类型，支持在线推理和批量推理。默认值为batch，即采用批量推理
                - online： 采用方舟平台提供的在线推理模块进行推理
                - batch：采用方舟平台提供的批量推理模块进行推理
            system_content: 系统提示内容
                系统提示内容，以system角色作为模型的输入
            prompt: 用户提示词，
                用户提示词，用于指导模型的行为。配置该字段时，会和输入的文本拼接，以user角色方式输入给模型。同时，该字段也可以配置为{query}，此时，输入的文本会替换掉该字段.
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

        self.system_content = system_content
        self.prompt = prompt

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=model)

    def transform(self, raw_text: pa.Array) -> pa.Array:
        """批量使用大模型对文本内容进行理解和回复.

        Args:
            raw_text: 包含待处理文本数据。类型为 str

        Returns:
            （默认情况下）当环境变量LAS_LLM_FINISH_REASON_CHECK=false时，返回字段类型为str。
            当环境变量LAS_LLM_FINISH_REASON_CHECK=true时，返回字段类型为struct，包含以下字段：
                - llm_result: 模型输出结果
                - finish_reason: 模型输出结束原因
        """
        messages = [gen_text_message(text, self.prompt, self.system_content) for text in raw_text.to_pylist()]
        return super().process(messages)
