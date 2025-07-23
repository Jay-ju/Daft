# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.repeated_lines_calculator import RepeatedLinesCalculator
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
        Name="重复行计算器",
        Clazz=RepeatedLinesCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["重复检测", "文本质量"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/repeated_lines_calculator/repeated_lines_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本中重复行的比例。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="第一行内容\n第二行内容\n第一行内容\n第三行内容",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="第一行内容\n第一行内容\n第一行内容",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="第一行内容\n\n第二行内容\n   \n第三行内容",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="0.25",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0.6666666666666666",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0.0",
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
