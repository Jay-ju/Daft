# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.multimodal.kimi_audio_understanding import KimiAudioUnderstanding
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


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="音频理解（Kimi-Audio 系列模型）",
        Clazz=KimiAudioUnderstanding,
        Category=Category.MULTI_MODAL,
        SubCategory=SubCategory.AUDIO_TO_TEXT,
        Tags=["音频理解", "多模态理解"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/kimi_audio_understanding/kimi_audio_understanding.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对音频进行理解和内容分析。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/kimi_audio_understanding/黑神话悟空对话.mp3",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="音频分析结果：'内容概括：这段音频是一位老人在讲述自己的经历和感受，内容涉及对过去生活的回忆和对未来的担忧...'",
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
