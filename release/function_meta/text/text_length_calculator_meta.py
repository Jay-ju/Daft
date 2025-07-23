# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.text_length_calculator import TextLengthCalculator
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
        Name="文本长度计算器",
        Clazz=TextLengthCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["文本长度", "字符统计"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/text_length_calculator/text_length_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本的字符长度。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello World",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="你好世界",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="Python编程 is fun",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="11",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="4",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="15",
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
