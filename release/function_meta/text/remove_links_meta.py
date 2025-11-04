# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.remove_links import RemoveLinks
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
        Name="文本链接移除",
        Clazz=RemoveLinks,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["移除链接", "文本处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/remove_links/remove_links.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子移除文本中的链接。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="prefix https://example.com/path suffix",
            Description="",
        )
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="prefix [LINK] suffix",
            Description="",
        )
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
