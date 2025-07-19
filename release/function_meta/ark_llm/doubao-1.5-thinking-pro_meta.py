# Copyright (c) Beijing Volcano Engine Technology Ltd.# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from function_meta.meta import (
    OP_BUCKET,
    OP_ENVIRONMENT,
    OP_REGION,
    OP_VERSION,
    Category,
    DataItem,
    ExtraMetaModel,
    OpMetaModel,
    SubCategory,
    ValueType,
)

from daft.las.functions.ark_llm.ark_llm_thinking_vision import ArkLLMThinkingVision


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="深度思考模型（Doubao-1.5-thinking-pro）",
        Clazz=ArkLLMThinkingVision,
        Category=Category.LLM_ONLINE_REASONING,
        SubCategory=SubCategory.TEXT_GENERATION,
        Tags=["文本生成", "深度思考"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doubao-1.5-thinking-pro/doubao-1.5-thinking-pro.py"
    code_description = (
        "下面的代码展示了如何使用 daft"
        "访问火山方舟 多模态深度思考模型（豆包系列） 模型进行批量推理。请注意每次大模型推理结果可能不同。"
    )
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/ark_llm_thinking_vision/eating_56.mp4",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""视频中呈现的是一段动画内容：起初展示的是一个**多层卡通风…""",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="""用户现在需要描述视频里的内容。首先看画面：开头是一个多层蛋…""",
            Description="",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
