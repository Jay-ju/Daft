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

from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="文本生成或视觉理解（豆包/DeepSeek 系列模型）",
        Clazz=ArkLLMGenerate,
        Description="提供通用的文本或视觉理解能力，输入字段格式需满足:[{'role': 'user', 'content': query}]类似的格式，"
        "详细格式请参考方舟大模型服务平台要求的messages格式（https://www.volcengine.com/docs/82379/1494384）",
        Category=Category.LLM_ONLINE_REASONING,
        SubCategory=SubCategory.VISION_TO_TEXT,
        Tags=["多模态"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/ark_llm_generate/ark_llm_generate.py"
    code_description = (
        "下面的代码展示了如何使用 daft"
        "访问火山方舟 文本生成或视觉理解模型进行批量推理。请注意每次大模型推理结果可能不同。"
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
