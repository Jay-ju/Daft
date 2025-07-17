# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioAsrDoubao
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
        Name="语音转文字（豆包语音大模型）",
        Clazz=AudioAsrDoubao,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_RECOGNITION,
        Tags=["语音识别", "ASR", "多语种"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_asr_doubao/audio_asr_doubao.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子将语音转换为文字。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_asr_doubao/sample.wav",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value='语音识别结果(原始)：\'{"audio_info":{"duration":3575},"result":{"text":"参观八达岭长城。"}}\'',
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="'说话人 1 0:00:00 0:00:02 \n参观八达岭长城。'",
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
