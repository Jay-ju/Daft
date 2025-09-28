# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.video.video_resize_resolution import VideoResizeResolution
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
        Name="视频分辨率调整",
        Clazz=VideoResizeResolution,
        Category=Category.VIDEO,
        SubCategory=SubCategory.VIDEO_PROCESSING,
        Tags=["视频处理", "分辨率调整", "尺寸控制", "视频编码", "批量处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_resize_resolution/video_resize_resolution.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子对视频进行智能分辨率调整。"
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_resize_resolution/sample.mp4",
            Description="待调整的原始视频",
        )
    ]
    after = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_resize_resolution/sample_resized.mp4",
            Description="分辨率调整后的视频",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
