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

from daft.las.functions.ark_llm.doubao_embedding_vision import DoubaoEmbeddingVision


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图文 embedding（豆包系列模型）",
        Clazz=DoubaoEmbeddingVision,
        Category=Category.LLM_ONLINE_REASONING,
        SubCategory=SubCategory.MULTI_MODAL_EMBEDDING,
        Tags=["图片向量化", "图文向量化", "图像向量化"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doubao_embedding_vision/doubao_embedding_vision.py"
    code_description = "下面的代码展示了如何使用 daft" "访问火山方舟图像向量化模型进行向量化计算。"
    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doubao_embedding_vision/cat_ip_adapter.jpeg",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="猫",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="[ 0.01916504  0.01879883  0.01135254 ... -0.01818848  0.04443359 -0.00714111]",
            Description="image embedding",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
