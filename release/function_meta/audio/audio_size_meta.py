# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioSize
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
        Name="音频文件大小计算",
        Clazz=AudioSize,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_PROCESSING,
        Tags=["元数据", "文件分析", "音频属性"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_size/audio_size.py"
    code_description = (
        "下面的代码展示了如何使用 pandas（适用于单机）" "和 ray（适用于分布式）运行算子计算音频文件大小。"
    )
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_size/sample.mp3",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="795426.0",
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
