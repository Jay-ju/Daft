# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.infra.las_ark import (
    DEFAULT_INFERENCE_TYPE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REQUEST_TIMEOUT,
    LasArkClient,
    LasArkConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_LAS_LLM_FINISH_REASON_CHECK = os.getenv("LAS_LLM_FINISH_REASON_CHECK", "false").lower() == "true"


class ArkLLMGenerate(Operator):
    _finish_reason_check = DEFAULT_LAS_LLM_FINISH_REASON_CHECK

    def __init__(
        self,
        model: str,
        version: str,
        access_key: str | None = None,
        account_id: str | None = None,
        inference_type: str = DEFAULT_INFERENCE_TYPE,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        logit_bias: dict[str, Any] | None = None,
        tools: list[dict[Any, Any]] | None = None,
        llm_config: dict[str, Any] | None = None,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        **kwargs: dict[str, Any],
    ) -> None:
        """使用豆包/DeepSeek模式做LLM推理.

        Args:
            model: 模型名称
                支持的模型有:豆包模型和DeepSeek模型。 示例 doubao-1.5-lite-32k
            version: 模型版本
                输入模型对应的版本信息。示例 250115
            access_key: 用户的ak
                用户的ak，用于鉴权，校验当前用户是否开通过LAS且有工作流白名单能力
            account_id: 账号ID
                账号ID，用于校验当前用户是否有模型权限
            inference_type: 推理类型，支持在线推理和批量推理。默认值为batch，即采用批量推理
                - online： 采用方舟平台提供的在线推理模块进行推理
                - batch：采用方舟平台提供的批量推理模块进行推理
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
        super().__init__(**kwargs)

        ark_config = LasArkConfig.from_env()
        ark_config.request_timeout = request_timeout
        ark_config.max_concurrency = max_concurrency
        ark_config.max_connections = max_concurrency
        ark_config.max_keepalive_connections = max(int(max_concurrency / 2), 1)
        ark_config.inference_type = inference_type
        self.client = LasArkClient(config=ark_config)

        self.access_key = access_key or ark_config.access_key
        self.account_id = account_id or ark_config.account_id
        self.llm_config = llm_config or {}

        options_tmp = {
            "model_name": model,
            "version": version,
            "access_key": self.access_key,
            "account_id": self.account_id,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "logit_bias": logit_bias,
            "tools": tools,
            "type": inference_type,
        }
        self.options = {k: v for k, v in options_tmp.items() if v is not None}

        self.options |= self.llm_config

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        if ArkLLMGenerate._finish_reason_check:
            return pa.struct({"llm_result": pa.string(), "finish_reason": pa.string()})
        return pa.string()

    def transform(self, messages: pa.Array) -> pa.Array:
        """批量使用大模型进行文本数组推理.

        该方法使用预加载的大模型对输入的文本数组进行批量推理，生成对应的模型输出结果。

        Args:
            messages: 包含待处理消息的PyArrow数组。类型为list[dict].
                该字段需要符合方舟大模型服务提供的chat API中messages字段的格式。
                messages格式可以参考https://www.volcengine.com/docs/82379/1494384

        Returns:
            pyarrow.Array: 处理后的PyArrow数组。若过程中有数据处理失败，返回与正常返回类型一致的空数组。
                当环境变量LAS_LLM_FINISH_REASON_CHECK=true时，返回字段类型为struct，包含以下字段：
                    - llm_result: 模型输出结果
                    - finish_reason: 模型输出结束原因
                当环境变量LAS_LLM_FINISH_REASON_CHECK=false时，返回字段类型为str。
        """
        messages = messages.to_pylist()
        return self.process(messages)

    def process(self, messages: list[list[dict[Any, Any]]]) -> pa.Array:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._async_transform(messages))

    async def _async_transform(self, messages: list[list[dict[Any, Any]]]) -> pa.Array:
        requests = [{"messages": msg, **self.options} for msg in messages]
        results = await self.client.batch_process(requests)
        output_array, finish_reason_array = self._update_array_with_results(results)
        if ArkLLMGenerate._finish_reason_check:
            return pa.StructArray.from_arrays([output_array, finish_reason_array], ["llm_result", "finish_reason"])

        return output_array

    def _update_array_with_results(self, results: list[dict[str, Any]]) -> tuple[pa.Array, pa.Array]:
        # init output_data and finish_reason_data
        output_data = [None] * len(results)
        finish_reason_data = [None] * len(results)

        for i, result in enumerate(results):
            output_data[i] = result.get("choices", [{}])[0].get("message", {}).get("content")

            if ArkLLMGenerate._finish_reason_check:
                if "error" in result:
                    finish_reason_data[i] = result.get("error")
                else:
                    finish_reason_data[i] = result.get("choices", [{}])[0].get("finish_reason")

        return (pa.array(output_data, type=pa.string()), pa.array(finish_reason_data, type=pa.string()))
