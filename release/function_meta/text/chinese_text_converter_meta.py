# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.chinese_text_converter import ChineseTextConverter
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
        Name="中文简繁体转换",
        Clazz=ChineseTextConverter,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["简繁转换", "中文处理", "OpenCC"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/chinese_text_converter/chinese_text_converter.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子进行中文简繁体转换。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""這是一個繁體中文的測試文本，包含了一些專業術語和技術名詞。""",
            Description="繁体中文文本示例",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="""Hello 世界！這裡有中英文混雜的內容，測試OpenCC是否能正確處理。""",
            Description="中英文混杂文本示例",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="这是一个繁体中文的测试文本，包含了一些专业术语和技术名词。",
            Description="转换后的简体中文文本",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello 世界！这里有中英文混杂的内容，测试OpenCC是否能正确处理。",
            Description="转换后的中英文混杂文本",
        ),
    ]
    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
