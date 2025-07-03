# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.video import VideoKeyframes
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
        Name="视频关键帧抽取",
        Clazz=VideoKeyframes,
        Category=Category.VIDEO,
        SubCategory=SubCategory.VIDEO_PROCESSING,
        Tags=["视频分析", "帧提取", "内容分析", "关键帧"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_keyframes/video_keyframes.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子抽取视频关键帧。"
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_keyframes/sample.mp4",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_keyframes/sample/keyframe_0001.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_keyframes/sample/keyframe_0002.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_keyframes/sample/keyframe_0003.jpg",
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
