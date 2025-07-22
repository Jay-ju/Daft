# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.copyright_cleaner import CopyrightCleaner
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
        Name="版权声明移除",
        Clazz=CopyrightCleaner,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["版权清理", "文本处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/copyright_cleaner/copyright_cleaner.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子移除文本中的版权声明内容。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="/* \n * Copyright (c) 2023 Jane Smith\n * 本代码依据 Apache License 2.0 授权，详见随附的 LICENSE 文件。\n */ 你好",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="# 版权 (c) 2023 Jane Smith\n# 本代码依据 Apache License 2.0 授权，详见随附的 LICENSE 文件。\n 你好",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value=" 你好",
            Description="",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value=" 你好",
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
