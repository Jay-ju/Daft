# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.bullet_line_ratio_calculator import BulletLineRatioCalculator
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
        Name="项目符号行占比计算器",
        Clazz=BulletLineRatioCalculator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["文本分析", "结构化检测"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/bullet_line_ratio_calculator/bullet_line_ratio_calculator.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算文本中项目符号行的占比。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="第一行内容\n• 项目符号行1\n- 项目符号行2\n第三行内容\n* 项目符号行3",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="0.6",
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
