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

from daft.las.functions.image.image_nsfw_detect import ImageNsfwDetect


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图片安全性检测",
        Clazz=ImageNsfwDetect,
        Category=Category.IMAGE,
        SubCategory=SubCategory.IMAGE_PROCESSING,
        Tags=["图片安全性检测", "图片处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_nsfw_detect/image_nsfw_detect.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对图像做安全性检测。"

    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_nsfw_detect/cat_ip_adapter.png",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="0.000114",
            Description="NSFW Score",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
