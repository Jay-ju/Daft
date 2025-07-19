# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioTtsDoubao
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
        Name="文字转语音（豆包语音大模型）",
        Clazz=AudioTtsDoubao,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_GENERATION,
        Tags=["语音合成", "TTS", "多语种"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_tts_doubao/audio_tts_doubao.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子将文字转换为语音。"

    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="今天天气真好，适合出去走走。",
            Description="",
        )
    ]

    after = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_tts_doubao/tts_result.mp3",
            Description="",
        )
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
