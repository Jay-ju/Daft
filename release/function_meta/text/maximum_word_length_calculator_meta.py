# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.maximum_word_length_calculator import MaximumWordLengthCalculator
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
        Name="最大英文单词长度计算器",
        Clazz=MaximumWordLengthCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["英文单词", "长度统计"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/maximum_word_length_calculator/maximum_word_length_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本中英文单词的最大长度。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello world 你好世界",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="Python编程 is fun",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="这是一个中文句子",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="5",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="6",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0",
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
