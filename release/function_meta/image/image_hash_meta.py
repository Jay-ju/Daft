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

from daft.las.functions.image.image_hash import ImageHash


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图片Hash值",
        Clazz=ImageHash,
        Category=Category.IMAGE,
        SubCategory=SubCategory.IMAGE_PROCESSING,
        Tags=["图片Hash值", "图片处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_hash/image_hash.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对图像做Hash值计算。"

    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_hash/cat_ip_adapter.png",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="8d3986a636e768ad",
            Description="hash_hex",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="1000110100111001100001101010011000110110111001110110100010101101",
            Description="hash_bin",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
