# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.video.video_extract_metadata import VideoExtractMetadata
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
        Name="视频元数据提取",
        Clazz=VideoExtractMetadata,
        Category=Category.VIDEO,
        SubCategory=SubCategory.VIDEO_PROCESSING,
        Tags=["元数据提取", "视频信息", "视频分析", "ffprobe", "视频元数据", "格式检测"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_extract_metadata/video_extract_metadata.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子提取视频文件的元数据信息。"
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_extract_metadata/music_sample.mp4",
            Description="示例视频文件",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Dict.name,
            Value='{"duration": 7.367, "format_name": "mov,mp4,m4a,3gp,3g2,mj2", "bit_rate": 1058326, "has_video": true, "has_audio": true, "video_codec": "h264", "video_width": 720, "video_height": 720, "video_fps": 30.0, "audio_codec": "aac", "audio_sample_rate": 44100, "audio_channels": 2}',
            Description="视频文件元数据：包含视频和音频流信息",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
