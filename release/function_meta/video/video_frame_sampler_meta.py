# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.video.video_frame_sampler import VideoFrameSampler
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
        Name="视频帧采样",
        Clazz=VideoFrameSampler,
        Category=Category.VIDEO,
        SubCategory=SubCategory.VIDEO_PROCESSING,
        Tags=["视频处理", "采样", "抽帧", "预处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/video_frame_sampler.py"
    code_description = (
        "下面的代码展示了如何使用 Daft（适用于分布式）从视频中按帧间隔采样若干图像帧。"
        "本示例设置 interval_frames=100，实际产出 6 张图片。"
    )
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample.mp4",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample/frame_0000.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample/frame_0001.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample/frame_0002.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample/frame_0003.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample/frame_0004.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample/frame_0005.jpg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_frame_sampler/sample/frame_0006.jpg",
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
