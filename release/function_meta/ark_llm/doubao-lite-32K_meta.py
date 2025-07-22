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

from daft.las.functions.ark_llm.ark_llm_text_generate import ArkLLMTextGenerate


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="文本生成（doubao-lite-32K）",
        Clazz=ArkLLMTextGenerate,
        Category=Category.LLM_ONLINE_REASONING,
        SubCategory=SubCategory.TEXT_GENERATION,
        Tags=["文本生成"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doubao-lite-32K/doubao-lite-32K.py"
    code_description = (
        "下面的代码展示了如何使用 daft" "访问火山方舟 文本生成 模型进行批量推理。" "请注意每次大模型推理结果可能不同。"
    )
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""中国的首都在哪里""",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="""十字花科植物有哪些""",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""中国的首都是北京。""",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="""白菜、萝卜、油菜、花椰菜、荠菜等都是十字花科植物。""",
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
