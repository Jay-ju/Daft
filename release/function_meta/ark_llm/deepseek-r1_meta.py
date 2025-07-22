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

from daft.las.functions.ark_llm.ark_llm_thinking_vision import ArkLLMThinkingVision


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="深度思考（deepseek-r1）",
        Clazz=ArkLLMThinkingVision,
        Category=Category.LLM_ONLINE_REASONING,
        SubCategory=SubCategory.TEXT_GENERATION,
        Tags=["文本生成", "深度思考"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/deepseek-r1/deepseek-r1.py"
    code_description = (
        "下面的代码展示了如何使用 daft" "访问火山方舟 文本生成 模型进行批量推理。" "请注意每次大模型推理结果可能不同。"
    )
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""帮我规划5月去新疆的10天旅行安排""",
            Description="",
        )
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""以下是为你规划的5月新疆10天旅行安排，主打北疆伊犁环…""",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="""好的，用户让我帮忙规划一个5月份去新疆的10天旅行安排。首…""",
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
