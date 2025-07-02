# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.embedding.bge_sparse_dense_embedding import BgeSparseDenseEmbedding
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
        Name="文本 sparse & dense embedding（BGE模型）",
        Clazz=BgeSparseDenseEmbedding,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_EMBEDDING,
        Tags=["文本嵌入", "BGE"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/bge_sparse_dense_embedding/bge_sparse_dense_embedding.py"
    code_description = "下面的代码展示了如何使 daft 运行算子基于bge-m3模型计算文本dense embedding、sparse embedding以及token embedding。"
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
            Value="[-0.041961669921875, 0.021759033203125, -0.032440185546875, 0.01076507568359375, -0.0188446044921875, -0.03759765625, -0.043487548828125, -0.05889892578125, ... , 0.031280517578125, -0.036041259765625, 0.0072479248046875]",
            Description="dense embedding",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="{'Hello': 0.31201171875, 'World': 0.317626953125, '!': 0.2178955078125}",
            Description="sparse embedding",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="[[0.006718197371810675, 0.044136520475149155, 0.014063426293432713, 0.022426564246416092, 0.017442883923649788, 0.02412036433815956, ... , 0.017784900963306427, 0.04674236848950386, -0.009356616996228695], "
            "[0.004435727838426828, 0.03958163410425186, 0.013823356479406357, 0.017340045422315598, -0.005102975759655237, 0.005405125208199024, ... , 0.02150299958884716, 0.046900372952222824, 0.016635028645396233], "
            "[0.014248124323785305, 0.027477290481328964, 0.029532475396990776, 0.04752828925848007, 0.03972204402089119, 0.030119670554995537, ..., -0.0074608358554542065, 0.0358189195394516, -0.013980432413518429], "
            "[0.04812050610780716, 0.01809948869049549, -0.007034667767584324, 0.02881438471376896, -0.019728440791368484, -0.021309129893779755, ... , 0.0010776736307889223, 0.03820198401808739, 0.0238068588078022]]",
            Description="tokens embedding",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
