# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

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

from daft.las.functions.image.image_aesthetic_score import ImageAestheticScore


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图像美学评分",
        Clazz=ImageAestheticScore,
        Category=Category.IMAGE,
        SubCategory=SubCategory.IMAGE_PROCESSING,
        Tags=["美学评分", "图像质量", "构图分析", "美学评估", "图像分析", "CLIP", "美学质量"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_aesthetic_score/image_aesthetic_score.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子对图像进行美学质量评分。"
    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_aesthetic_score/high_quality_landscape.jpg",
            Description="高质量风景图像",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_aesthetic_score/portrait_photo.jpg",
            Description="人像摄影图像",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_aesthetic_score/low_quality_blur.jpg",
            Description="低质量模糊图像",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="0.852",
            Description="美学评分：0.852（高质量）",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0.734",
            Description="美学评分：0.734（中等质量）",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0.234",
            Description="美学评分：0.234（低质量）",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
