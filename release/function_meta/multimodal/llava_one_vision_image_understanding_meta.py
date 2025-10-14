# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.multimodal.llava_one_vision_image_understanding import LlavaOneVisionImageUnderstanding
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
        Name="图片内容理解（LLaVA 系列模型）",
        Clazz=LlavaOneVisionImageUnderstanding,
        Category=Category.MULTI_MODAL,
        SubCategory=SubCategory.IMAGE_TO_TEXT,
        Tags=["图片理解", "多模态理解", "LLaVA"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/llava_one_vision_image_understanding/llava_one_vision_image_understanding.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子理解图像内容，并按照指令生成描述。"
    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/llava_one_vision_image_understanding/cat_ip_adapter.jpeg",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="这幅图片属于动画类型，具体来说是CGI（计算机生成图像）动画。这一分类是基于角色的风格和制作质量，这些特征与传统动画不同，后者通常具有更明显的线条和纹理。CGI动画以其逼真的外观而闻名，可以实现高度详细的角色和环境，就像在这张图片中所见。这种技术常用于电影、电视节目和视频游戏中，以创造视觉上引人入胜且栩栩如生的场景。",
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
