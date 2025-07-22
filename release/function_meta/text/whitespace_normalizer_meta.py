# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.whitespace_normalizer import WhitespaceNormalizer
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
        Name="空白字符标准化器",
        Clazz=WhitespaceNormalizer,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["文本清洗", "格式标准化"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/whitespace_normalizer/whitespace_normalizer.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子标准化文本中的空白字符。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello\u2000World\u2001Test",
            Description="包含Unicode空白字符的文本",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello World Test",
            Description="标准化后的文本",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
