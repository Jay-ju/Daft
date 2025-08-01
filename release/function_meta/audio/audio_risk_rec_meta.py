# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import json

from daft.las.functions.audio import AudioRiskRec
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
        Name="音频风险识别",
        Clazz=AudioRiskRec,
        Category=Category.AUDIO,
        SubCategory=SubCategory.AUDIO_CONTENT_SECURITY,
        Tags=["风险检测", "内容审核", "音频"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_risk_rec/audio_risk_rec.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子进行音频内容风险识别。"
    before = [
        DataItem(
            Type=ValueType.Audio.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/audio_risk_rec/sample.mp3",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Dict.name,
            Value=json.dumps({"Decision": "PASS", "Message": "success"}, indent=4, ensure_ascii=False),
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
