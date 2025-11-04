# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.embedding.bge_embedding import BgeEmbedding
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
        Name="文本 embedding（BGE模型）",
        Clazz=BgeEmbedding,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_EMBEDDING,
        Tags=["文本嵌入", "BGE"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/bge_embedding/bge_embedding.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子基于 bge-m3 模型计算文本embedding。"
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
            Value="[-0.04205322  0.02178955 -0.0324707   0.01073456 -0.01882935 -0.03759766, ... , 0.031311035, -0.036071777, 0.0072364807]",
            Description="dense embedding",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
