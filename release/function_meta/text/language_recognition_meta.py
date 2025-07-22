# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.language_recognition import LanguageRecognitionOperator
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
        Name="文本语种识别",
        Clazz=LanguageRecognitionOperator,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_CLASSIFICATION,
        Tags=["语种识别", "多语言", "fasttext"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/language_recognition/language_recognition.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子识别文本的语种。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="这是一行测试内容。",
            Description="中文文本",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="This is a test content.",
            Description="英文文本",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="こんにちは",
            Description="日文文本",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="안녕하세요",
            Description="韩文文本",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Dict.name,
            Value='{"language": "zh", "confidence": 1.000048279762268}',
            Description="中文识别结果",
        ),
        DataItem(
            Type=ValueType.Dict.name,
            Value='{"language": "en", "confidence": 0.9209088683128357}',
            Description="英文识别结果",
        ),
        DataItem(
            Type=ValueType.Dict.name,
            Value='{"language": "ja", "confidence": 1.0000269412994385}',
            Description="日文识别结果",
        ),
        DataItem(
            Type=ValueType.Dict.name,
            Value='{"language": "ko", "confidence": 0.9996028542518616}',
            Description="韩文识别结果",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
