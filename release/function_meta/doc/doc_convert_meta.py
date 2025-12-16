# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.doc.doc_convert import DocConvert
from release.function_meta.meta import (
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
        Name="Doc格式转换",
        Clazz=DocConvert,
        Category=Category.DOC,
        SubCategory=SubCategory.DOC_PARSE,
        Tags=["doc", "docx", "pdf", "office", "格式转化"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doc_convert/doc_convert.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）执行 docx 转 pdf。"
    before = [
        DataItem(
            Type=ValueType.File.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doc_convert/sample.docx",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.File.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/doc_convert/sample.pdf",
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
