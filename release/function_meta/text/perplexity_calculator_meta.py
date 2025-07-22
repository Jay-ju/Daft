# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.perplexity_calculator import PerplexityCalculator
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
        Name="困惑度计算器",
        Clazz=PerplexityCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["文本质量评估", "困惑度计算"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/perplexity_calculator/perplexity_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本的困惑度，用于评估文本质量。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="人工智能技术正在快速发展，人工智能技术已经广泛应用于各个领域。",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="乱码文本 12345 !@#$%",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Float.name,
            Value=335.9,
            Description="中文文本困惑度",
        ),
        DataItem(
            Type=ValueType.Float.name,
            Value=274.0,
            Description="英文文本困惑度",
        ),
        DataItem(
            Type=ValueType.Float.name,
            Value=9081.4,
            Description="乱码文本困惑度",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
