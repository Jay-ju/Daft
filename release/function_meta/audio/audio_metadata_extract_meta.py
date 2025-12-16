# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio.audio_metadata_extract import AudioMetadataExtract
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
        Name="音频元数据提取",
        Clazz=AudioMetadataExtract,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_PROCESSING,
        Tags=["元数据提取", "音频信息", "音频分析", "ffprobe", "音频元数据", "格式检测"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_metadata_extract/audio_metadata_extract.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子提取音频文件的元数据信息。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_metadata_extract/sample.wav",
            Description="示例音频文件",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Dict.name,
            Value='{"duration": 49.71102, "format_name": "wav", "bit_rate": 1411212, "audio_codec": "pcm_s16le", "audio_sample_rate": 44100, "audio_channels": 2}',
            Description="音频文件元数据：包含音频流信息",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
