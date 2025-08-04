# Copyright (c) Beijing Volcano Engine Technology Ltd.# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

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

from daft.las.functions.ark_llm.doubao_embedding_text import DoubaoEmbeddingText


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="文本向量化（Doubao-embedding）",
        Clazz=DoubaoEmbeddingText,
        Category=Category.LLM_ONLINE_REASONING,
        SubCategory=SubCategory.TEXT_EMBEDDING,
        Tags=["文本向量化"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doubao-embedding/doubao-embedding.py"
    code_description = (
        "下面的代码展示了如何使用 daft"
        "访问火山方舟 文本向量化 模型进行批量推理。"
        "请注意每次大模型推理结果可能不同。"
    )
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""Hello World!""",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""[0.080078125, 1.8359375, 0.84…""",
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
