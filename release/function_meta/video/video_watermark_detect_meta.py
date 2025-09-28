# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.video.video_watermark_detect import VideoWatermarkDetect
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
        Name="视频水印检测",
        Clazz=VideoWatermarkDetect,
        Category=Category.VIDEO,
        SubCategory=SubCategory.VIDEO_PROCESSING,
        Tags=["水印检测", "文本识别", "OCR", "视频分析", "区域定位"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_watermark_detect/video_watermark_detect.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子对视频进行水印检测。"
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/video_watermark_detect/sample.mp4",
            Description="",
        )
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="检测结果: {'watermark_regions': [{'ymin': 85, 'ymax': 179, 'xmin': 1497, 'xmax': 1856, 'confidence': 1.0}], 'video_resolution': [1920, 1080], 'total_frames': 5854}",
            Description="水印检测结果",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
