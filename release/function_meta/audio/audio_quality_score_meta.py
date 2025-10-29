# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioQualityScore
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
        Name="音频质量评分(DNSMOS)",
        Clazz=AudioQualityScore,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_PROCESSING,
        Tags=["音频处理", "质量评分", "DNSMOS", "质量评估"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_quality_score/audio_quality_score.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对音频进行质量评分。"

    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_quality_score/test_audio.wav",
            Description="",
        )
    ]

    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="质量评分: 3.85",
            Description="音频质量评分范围在 1.0 到 5.0 之间，分数越高表示音频质量越好",
        )
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
