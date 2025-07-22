# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

from daft.las.functions.doc.parse.xlsx_parse import XlsxParse
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
        Name="Xlsx 文档解析",
        Clazz=XlsxParse,
        Category=Category.DOC,
        SubCategory=SubCategory.DOC_PARSE,
        Tags=["excel", "表格处理", "数据解析", "office"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/xlsx_parse/xlsx_parse.py"
    code_description = "下面的代码展示了如何使用 pandas（适用于单机）" "和 ray（适用于分布式）运行算子解析xlsx文档。"
    before = [
        DataItem(
            Type=ValueType.File.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/xlsx_parse/sample.xlsx",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="|   产品ID | 产品名称   |   价格 |   库存 |\n|---------:|:-----------|-------:|-------:|\n|     1001 | 笔记本电脑 |   5999 |     15 |\n|     1002 | 智能手机   |   3999 |     30 |\n|     1003 | 耳机       |    299 |     50 |",
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
