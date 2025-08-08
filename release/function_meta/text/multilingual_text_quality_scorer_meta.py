# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.multilingual_text_quality_scorer import MultilingualTextQualityScorer
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
        Name="多语言文本质量评分",
        Clazz=MultilingualTextQualityScorer,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_QUALITY_ASSESSMENT,
        Tags=["文本质量", "多语言", "E5"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/multilingual_text_quality_scorer/multilingual_text_quality_scorer.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子基于E5模型对多语言文本质量进行评分。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""这是一篇关于人工智能技术发展的高质量学术论文，内容详实且具有很强的参考价值。""",
            Description="中文文本示例",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="""これは量子物理学とその現代技術への応用に関するよく書かれた科学論文です。""",
            Description="日文文本示例",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="0.6294931",
            Description="中文文本质量分数（0-1范围，分数越高质量越好）",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="0.73835784",
            Description="日文文本质量分数（0-1范围，分数越高质量越好）",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
