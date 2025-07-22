# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.special_characters_ratio_calculator import SpecialCharactersRatioCalculator
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
        Name="特殊字符占比计算器",
        Clazz=SpecialCharactersRatioCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["特殊字符统计", "文本特征"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/special_characters_ratio_calculator/special_characters_ratio_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本中特殊字符的占比。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello world!",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="1234567890",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Float.name,
            Value=0.16666666666666666,
            Description="特殊字符占比：2/12 ≈ 0.167（感叹号+空格）",
        ),
        DataItem(
            Type=ValueType.Float.name,
            Value=1.0,
            Description="特殊字符占比：10/10 = 1.0（全数字）",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
