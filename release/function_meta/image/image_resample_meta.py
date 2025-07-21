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

from daft.las.functions.image.image_resample import ImageResample


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图片重采样",
        Clazz=ImageResample,
        Category=Category.IMAGE,
        SubCategory=SubCategory.IMAGE_PROCESSING,
        Tags=["图片重采样", "图片处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_resample/image_resample.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对图像做重采样。"

    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_resample/cat_ip_adapter.png",
            Description="image size: 2000 x 2000",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_resample/cat_ip_adapter_resample.jpg",
            Description="image size: 200 x 200",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
