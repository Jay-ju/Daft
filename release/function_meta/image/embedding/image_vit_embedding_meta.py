# Copyright (c) Beijing Volcano Engine Technology Ltd.

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

from daft.las.functions.image.embedding.image_vit_embedding import ImageViTEmbedding


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图像 Embedding（ViT 系列模型）",
        Clazz=ImageViTEmbedding,
        Category=Category.IMAGE,
        SubCategory=SubCategory.IMAGE_EMBEDDING,
        Tags=["图像嵌入", "图像语义"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_vit_embedding/image_vit_embedding.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子计算图片的 embedding。"
    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/image_vit_embedding/cat_ip_adapter.png",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="[-0.01072166, -0.01862401, 0.02559144, 0.03767537, 0.01172297, -0.01271369, ... , -0.08253827, -0.03676888, 0.02016414]",
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
