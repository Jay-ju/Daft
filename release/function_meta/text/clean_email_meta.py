# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.clean_email import CleanEmail
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
        Name="email 邮箱清理",
        Clazz=CleanEmail,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_CLEAN,
        Tags=["文本清洗", "email 邮箱"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/clean_email/clean_email.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对 email 邮箱进行移除。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""lihua@163.com This is a test content.""",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="This is a test content.",
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
