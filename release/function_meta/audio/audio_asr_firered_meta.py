# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.audio import AudioAsrFireRed
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
        Name="语音转文字（FireRed）",
        Clazz=AudioAsrFireRed,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_RECOGNITION,
        Tags=["语音识别", "ASR", "多语种", "FireRed"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_asr_firered/audio_asr_firered.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子将语音转换为文字。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_asr_firered/sample_normal.wav",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="人我保住了金我取到了俺老孙啥功名不要只求回到这花果山终老过过逍遥日子上面的天王老子信不过我我懂让你小子带些虾兵蟹将过来虚张声势又想唬我回去做神仙嘿嘿嘿嘿我也懂我不懂的是你他娘的杀我猴子猴孙",
            Description="AED模型识别结果",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="人我保住了金我取到了俺老孙啥功名不要只求回到这花果山中了过过逍遥日子上面的天王老子信不过我我懂让你小子带些虾兵蟹将过来虚张声势又想唬我回去做神仙我也懂我不懂的是你他娘的杀我猴子猴孙",
            Description="LLM模型识别结果",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
