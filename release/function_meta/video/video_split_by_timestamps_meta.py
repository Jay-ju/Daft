# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.video import VideoSplitByTimestamps
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
        Name="视频片段切分(时间戳)",
        Clazz=VideoSplitByTimestamps,
        Category=Category.VIDEO,
        SubCategory=SubCategory.VIDEO_PROCESSING,
        Tags=["视频编辑", "分割", "时间戳", "剪辑"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_split_by_timestamps/video_split_by_timestamps.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子按时间戳切分视频。"
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_split_by_timestamps/sample.mp4",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_split_by_timestamps/sample/segment_0_0-2_0.mp4",
            Description="",
        ),
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_split_by_timestamps/sample/segment_2_0-4_0.mp4",
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
