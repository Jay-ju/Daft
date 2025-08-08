# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.text_safety_scorer import TextSafetyScorer
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
        Name="文本安全性评分器",
        Clazz=TextSafetyScorer,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["文本安全", "内容审核", "风险评估", "多语言", "ShieldLM"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/text_safety_scorer/text_safety_scorer.py"
    code_description = (
        "下面的代码展示了如何使用 daft 运行算子评估文本安全性，输出safe、unsafe、controversial三分类概率。"
    )
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="爱与和平是世界的主旋律。",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="我讨厌所有人，最好都去死。",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value='{"safe": 0.8632, "unsafe": 0.0443, "controversial": 0.0924}',
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value='{"safe": 0.0005, "unsafe": 0.9994, "controversial": 0.0001}',
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
