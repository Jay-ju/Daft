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

from daft.las.functions.image.image_easyocr import ImageEasyOcr


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图像 OCR（EasyOCR）",
        Clazz=ImageEasyOcr,
        Category=Category.IMAGE,
        SubCategory=SubCategory.IMAGE_OCR,
        Tags=["图像OCR", "OCR", "多语言"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_easyocr/image_easyocr.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子识别图像中的文字。"

    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_easyocr/通用场景图片.jpeg",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="不论结局\n我己经很感谢相遇",
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
