# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio.audio_ffmpeg_wrapped import AudioFFMPEGWrapped
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
        Name="音频滤镜处理",
        Clazz=AudioFFMPEGWrapped,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_PROCESSING,
        Tags=["音频处理", "滤镜", "FFmpeg", "降噪", "均衡"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_ffmpeg_wrapped/audio_ffmpeg_wrapped.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子对音频应用 FFmpeg 滤镜。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_ffmpeg_wrapped/sample.wav",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_ffmpeg_wrapped/sample_processed.wav",
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
