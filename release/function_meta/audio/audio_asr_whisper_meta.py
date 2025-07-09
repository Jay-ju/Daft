# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioAsrWhisper
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
        Name="语音转文字（whisper 系列模型）",
        Clazz=AudioAsrWhisper,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_RECOGNITION,
        Tags=["语音识别", "ASR", "多语种"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_asr_whisper/audio_asr_whisper.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子将语音转换为文字。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_asr_whisper/参观八达岭长城。.wav",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="语音识别结果：'参观八道岭长城'",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="分段语音识别结果：['参观八道岭长城']",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="分段时间戳：[[0.0, 3.56]]",
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
