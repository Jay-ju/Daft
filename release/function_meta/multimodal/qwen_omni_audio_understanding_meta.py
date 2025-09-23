# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.multimodal.qwen_omni_audio_understanding import QwenOmniAudioUnderstanding
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
        Name="音频内容理解（Qwen Omni 模型）",
        Clazz=QwenOmniAudioUnderstanding,
        Category=Category.MULTI_MODAL,
        SubCategory=SubCategory.AUDIO_TO_TEXT,
        Tags=["音频理解"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/qwen_omni_audio_understanding/qwen_omni_audio_understanding.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子理解音频内容，并按照指令生成描述。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/qwen_omni_audio_understanding/sample.mp3",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="人我保住了，经我取到了。俺老孙啥功名不要，只求回到这花果山中了，过过逍遥日子。上面的天王老子信不过我，我懂。敢用小子带些虾兵蟹将过来虚张声势，又想唬我回去做神仙。嘿嘿，我也懂。我不懂的是，你他娘的杀我猴子猴孙。",
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
