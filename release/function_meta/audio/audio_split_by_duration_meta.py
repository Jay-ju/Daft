# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioSplitByDuration
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
        Name="音频片段切分(时长)",
        Clazz=AudioSplitByDuration,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_PROCESSING,
        Tags=["音频编辑", "分割", "定长切割", "剪辑"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_duration/audio_split_by_duration.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子对音频按时长切分。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_duration/sample.mp3",
            Description="",
        )
    ]
    after = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_duration/sample/segment_1.mp3",
            Description="",
        ),
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_duration/sample/segment_2.mp3",
            Description="",
        ),
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_duration/sample/segment_3.mp3",
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
