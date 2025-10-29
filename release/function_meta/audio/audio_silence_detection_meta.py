# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio.audio_silence_detection import AudioSilenceDetection
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
        Name="音频静音检测",
        Clazz=AudioSilenceDetection,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_ANALYSIS,
        Tags=["静音检测", "音频分析", "质量检测", "音频处理", "静音识别", "ffmpeg", "volumedetect"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_silence_detection/audio_silence_detection.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子对音频进行静音检测。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_silence_detection/sample_silence.wav",
            Description="静音音频样本",
        ),
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_silence_detection/sample_speech.wav",
            Description="语音音频样本",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="true",
            Description="静音检测结果：true表示静音，false表示非静音",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="false",
            Description="静音检测结果：true表示静音，false表示非静音",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
