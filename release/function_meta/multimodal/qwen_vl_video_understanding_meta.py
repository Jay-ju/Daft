# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.multimodal.qwen_vl_video_understanding import QwenVLVideoUnderstanding
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
        Name="视频内容理解（Qwen VL 系列模型）",
        Clazz=QwenVLVideoUnderstanding,
        Category=Category.MULTI_MODAL,
        SubCategory=SubCategory.VIDEO_TO_TEXT,
        Tags=["视频理解", "多模态理解"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/qwen_vl_video_understanding/qwen_vl_video_understanding.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子理解视频内容，并按照指令生成描述。"
    before = [
        DataItem(
            Type=ValueType.Video.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/qwen_vl_video_understanding/eating_56.mp4",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="这是一段动画片段，画面中出现了一个卡通角色，它被设计成一个蛋糕的样子。这个角色有绿色的皮肤，眼睛和嘴巴都位于蛋糕的顶部，表情看起来有些生气或不满。蛋糕上有红色的奶油装饰，并且顶部还放着一颗红色的樱桃。背景是一个紫色的墙壁，墙上有一些装饰物，包括一个类似鱼的图案。整个场景充满了卡通风格，色彩鲜艳，给人一种活泼的感觉。",
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
