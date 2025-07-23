# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.commoncrawl_content_extractor import CommonCrawlContentExtractor
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
        Name="CommonCrawl WARC文件内容提取",
        Clazz=CommonCrawlContentExtractor,
        Description="从CommonCrawl的WARC文件中提取网页正文内容，支持多种解析策略和输入格式",
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["WARC提取", "CommonCrawl", "网页内容", "多格式支持"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/commoncrawl_content_extractor/commoncrawl_content_extractor.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子从CommonCrawl的WARC文件中提取网页正文，支持文件路径、二进制数据和base64编码等多种输入格式。"

    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="/path/to/warc/sample.warc.gz",
            Description="TOS挂载的WARC文件路径",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="[binary WARC data]",
            Description="二进制WARC数据",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value="data:application/octet-stream;base64,UklGRiQAAABXQVZFZm10...",
            Description="base64编码的WARC数据",
        ),
    ]

    after = [
        DataItem(
            Type=ValueType.List.name,
            Value=[
                {
                    "url": "http://example.com",
                    "content": "Hello World This is a test page about artificial intelligence.",
                    "warc_file": "sample.warc.gz",
                    "extractor": "trafilatura",
                },
                {
                    "url": "http://example.com",
                    "content": "Hello World This is a test page about artificial intelligence.",
                    "warc_file": "[binary_input]",
                    "extractor": "trafilatura",
                },
                {
                    "url": "http://example.com",
                    "content": "Hello World This is a test page about artificial intelligence.",
                    "warc_file": "data:application/octet-stream;base64,UklGRiQAAABXQVZFZm10...",
                    "extractor": "trafilatura",
                },
            ],
            Description="提取的网页内容列表，支持多种输入格式",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
