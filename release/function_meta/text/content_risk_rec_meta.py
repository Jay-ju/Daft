# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import json

from daft.las.functions.text import ContentRiskRec
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
        Name="文本内容风险识别",
        Clazz=ContentRiskRec,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_CONTENT_SECURITY,
        Tags=["风险检测", "内容审核"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/text/content_risk_rec.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子进行文本内容风险识别。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="今天天气真好，我们一起去公园玩吧！",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="出售枪支，联系电话123456789",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Dict.name,
            Value=json.dumps({"FinalLabel": "", "Decision": "PASS", "Message": "success"}, indent=4),
            Description="",
        ),
        DataItem(
            Type=ValueType.Dict.name,
            Value=json.dumps({"FinalLabel": "106", "Decision": "BLOCK", "Message": "success"}, indent=4),
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
