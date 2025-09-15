# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.multilingual_text_translate import MultilingualTextTranslate
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
        Name="多语言文本翻译",
        Clazz=MultilingualTextTranslate,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_TRANSLATION,
        Tags=["文本翻译", "多语言", "Doubao"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/multilingual_text_translate/multilingual_text_translate.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子基于Doubao模型对多语言文本进行翻译。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""这是一篇关于人工智能技术发展的高质量学术论文，内容详实且具有很强的参考价值。""",
            Description="中文文本原文示例",
        )
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""This is a high-quality academic paper on the development of artificial intelligence technology, which is detailed and has strong reference value. """,
            Description="中文文本翻译示例",
        )
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
