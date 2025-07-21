# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.regex_replacement import RegexReplacer
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
        Name="特定字符替换",
        Clazz=RegexReplacer,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_CLEAN,
        Tags=["文本清洗", "正则表达式"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/regex_replacement/regex_replacement.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对特定表达式进行替换。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""<Query></Query><Title>提升党的领导力，推进国家治理体系和治理能力现代化</Title><Url>http://news.cnr.cn/native/gd/20191212/t20191212_524895559.shtml</Url>""",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="/replace_tag/replace_tag/replace_tag提升党的领导力，推进中国治理体系和治理能力现代化/replace_tag/replace_taghttp://news.cnr.cn/native/gd/20191212/t20191212_524895559.shtml/replace_tag",
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
