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

from daft.las.functions.ark_llm.ark_llm_vision_understanding import ArkLLMVisionUnderstanding


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="视觉内容理解（Doubao-1.5-vision-pro）",
        Clazz=ArkLLMVisionUnderstanding,
        Category=Category.LLM_ONLINE_REASONING,
        SubCategory=SubCategory.VISION_TO_TEXT,
        Tags=["图片理解", "视频理解", "多模态理解"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doubao-1.5-vision-pro/doubao-1.5-vision-pro.py"
    code_description = (
        "下面的代码展示了如何使用 daft" "访问火山方舟 视觉理解 模型进行批量推理。" "请注意每次大模型推理结果可能不同。"
    )
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/ark_llm_vision_understanding/eating_56.mp4",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""视频中展示了一个卡通风格的场景。首先，画面中央是一个装饰有…""",
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
