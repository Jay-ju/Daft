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
        Name="文本生成",
        Clazz=ArkLLMTextGenerate,
        Category=Category.TEXT,
        SubCategory=SubCategory.LLM_ONLINE_REASONING,
        Description="针对纯文本的数据，调用方舟模型进行文本进行理解和回复。示例文本翻译、内容总结等场景，传入文本信息，通过大模型对这些文本信息作相应理解和回复."
        "输入纯文本数据，将其按照方舟模型的输入格式进行组装message信息：{role: user, content: <query语句>}。您只需要传入<query语句>即可.",
        Tags=["文本生成"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/ark_llm_text_generate/ark_llm_text_generate.py"
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
