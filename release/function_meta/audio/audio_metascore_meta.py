# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioMetascore
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
        Name="音频评分(Audiobox Aesthetics)",
        Clazz=AudioMetascore,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_PROCESSING,
        Tags=["音频处理", "质量评分", "音频美学", "质量评估", "audiobox_aesthetics"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_metascore/audio_metascore.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对音频进行评分。"

    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_metascore/sample.wav",
            Description="",
        )
    ]

    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="CE: 5.9, CU: 6.2, PC: 5.6, PQ: 7.4",
            Description="音频评分包含四个维度：CE(连贯性/听感投入度), CU(清晰度/可懂度), PC(制作质量/构成质量), PQ(感知质量/主观音质)",
        )
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
