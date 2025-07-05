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

from daft.las.functions.multimodal.embedding.clip_embedding import ClipEmbedding


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="图文 embedding（CLIP 模型）",
        Clazz=ClipEmbedding,
        Category=Category.MULTI_MODAL,
        SubCategory=SubCategory.MULTI_MODAL_EMBEDDING,
        Tags=["图文多模态", "图文向量化"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/clip_embedding/clip_embedding.py"
    code_description = "下面的代码展示了如何使用 daft 运行图文 embedding 算子， 生成图文 embedding 。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="小猫",
            Description="",
        ),
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/clip_embedding/cat_ip_adapter.jpeg",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="[1.26708984e-01, 1.55334473e-02, 6.07299805e-03, 9.24682617e-03, 2.08854675e-03, 3.20739746e-02, 1.44577026e-02, 3.51333618e-03, -4.18472290e-03, ... , -3.52096558e-03, -2.66571045e-02, -3.97338867e-02]",
            Description="text embedding",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="[4.59899902e-02, -9.01489258e-02, -2.70690918e-02,  1.37710571e-02, -4.27856445e-02, 1.29852295e-02, 4.90570068e-03, 2.29187012e-02, 2.03247070e-02, ... , 6.51931763e-03, 1.80358887e-02, -1.04808807e-03]",
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
