# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioSplitByTimestamps
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
        Name="音频片段切分(时间戳)",
        Clazz=AudioSplitByTimestamps,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_PROCESSING,
        Tags=["音频编辑", "分割", "时间点切割", "剪辑"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_timestamps/audio_split_by_timestamps.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子对音频按时间戳切分。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_timestamps/sample.mp3",
            Description="",
        )
    ]
    after = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_timestamps/sample/segment_0_0-5_0.mp3",
            Description="",
        ),
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_timestamps/sample/segment_5_0-10_0.mp3",
            Description="",
        ),
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_split_by_timestamps/sample/segment_10_0-15_0.mp3",
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
