# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.en_text_quality_scorer import EnTextQualityScorer
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
        Name="英文文本质量评分",
        Clazz=EnTextQualityScorer,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_QUALITY_ASSESSMENT,
        Tags=["文本质量", "FastText"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/en_text_quality_scorer/en_text_quality_scorer.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子基于FastText模型对英文文本质量进行评分。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""This is a well-written scientific article about quantum physics and its applications in modern technology.""",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="0.6918241381645203",
            Description="质量分数（0-2范围，分数越高质量越好）",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
