# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.word_repetition_calculator import WordRepetitionCalculator
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
        Name="词重复比例计算器",
        Clazz=WordRepetitionCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["文本分析", "重复检测"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/word_repetition_calculator/word_repetition_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本中基于N-gram的词组重复比例。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="人工智能技术正在快速发展，人工智能技术在各领域的应用越来越广泛。人工智能技术可以帮助我们解决复杂问题，人工智能技术的未来充满无限可能。",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Float.name,
            Value=0.14285714285714285,
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
