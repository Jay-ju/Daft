# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.url_ratio_calculator import UrlRatioCalculator
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
        Name="URL占比计算器",
        Clazz=UrlRatioCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["URL统计", "文本特征"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/url_ratio_calculator/url_ratio_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本中URL字符的占比。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello world",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="Check out https://example.com for more info",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="Visit http://test.com and https://demo.org",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="0.0",
            Description="URL字符占比：0/11 = 0.0",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0.4418604651162791",
            Description="URL字符占比：19/43 ≈ 0.442",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0.7380952380952381",
            Description="URL字符占比：31/42 ≈ 0.738",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
