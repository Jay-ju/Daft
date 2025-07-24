# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.pre_sign_url_for_tos import PreSignUrlForTos
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
        Name="生成TOS的普通预签名",
        Clazz=PreSignUrlForTos,
        Category=Category.TEXT,
        SubCategory=SubCategory.TOS_PRE_SIGN,
        Description="生成 TOS 文件路径签名 URL。当路径 schema 是 http 或 https 时，直接返回路径；若是 tos 或 s3，则对路径进行签名，其他情况返回 None。",
        Tags=["TOS 签名"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/pre_sign_url_for_tos/pre_sign_url_for_tos.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子，生成TOS的普通预签名。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="tos://tos_bucket/sample.mp4",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="https://tos_bucket.tos-cn-beijing.volces.com/sample…",
            Description="签名路径，实际情况为准，以https开头",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
