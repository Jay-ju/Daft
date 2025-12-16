# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.video.video_detect_audio import VideoDetectAudio
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
        Name="视频音频检测",
        Clazz=VideoDetectAudio,
        Category=Category.VIDEO,
        SubCategory=SubCategory.VIDEO_PROCESSING,
        Tags=["音频检测", "视频分析", "音频流", "音轨检测", "ffprobe"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_detect_audio/video_detect_audio.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子检测视频中是否存在音频。"
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_detect_audio/music_sample.mp4",
            Description="包含音频的视频",
        ),
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_detect_audio/music_sample_no_audio.mp4",
            Description="不包含音频的视频",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="true",
            Description="检测结果：存在音频",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="false",
            Description="检测结果：不存在音频",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
