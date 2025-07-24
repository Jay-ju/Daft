# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import json

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
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value=json.dumps(
                {
                    "url": "http://example.com",
                    "content": "This is the first page content.",
                    "warc_file": "sample.warc.gz",
                    "extractor": "trafilatura",
                },
                ensure_ascii=False,
                indent=2,
            ),
            Description="提取的第一页网页内容",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value=json.dumps(
                {
                    "url": "http://example2.com",
                    "content": "This is the second page content.",
                    "warc_file": "sample.warc.gz",
                    "extractor": "trafilatura",
                },
                ensure_ascii=False,
                indent=2,
            ),
            Description="提取的第二页网页内容",
        ),
        DataItem(
            Type=ValueType.Text.name,
            Value=json.dumps(
                {
                    "url": "http://example3.com",
                    "content": "This is the third page content.",
                    "warc_file": "sample.warc.gz",
                    "extractor": "trafilatura",
                },
                ensure_ascii=False,
                indent=2,
            ),
            Description="提取的第三页网页内容",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
