# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.alphanumeric_ratio_calculator import AlphanumericRatioCalculator
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
        Name="字符占比计算器",
        Clazz=AlphanumericRatioCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["字符统计", "文本特征"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/alphanumeric_ratio_calculator/alphanumeric_ratio_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本中字母和数字字符的占比。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="HelloWorld123",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello, world!",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="!!!@@@###$$$",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Float.name,
            Value=1.0,
            Description="字母数字字符占比：13/13 = 1.0",
        ),
        DataItem(
            Type=ValueType.Float.name,
            Value=0.7692307692307693,
            Description="字母数字字符占比：10/13 ≈ 0.769",
        ),
        DataItem(
            Type=ValueType.Float.name,
            Value=0.0,
            Description="字母数字字符占比：0/12 = 0.0",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
