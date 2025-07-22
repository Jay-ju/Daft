from __future__ import annotations

import base64
import os

import daft
from daft import col
from daft.las.functions.text.commoncrawl_content_extractor import CommonCrawlContentExtractor
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "ceshi/las_data/test/data")

    sample_warc_content = b"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: http://example.com\r\nContent-Type: application/http; msgtype=response\r\nContent-Length: 200\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body><h1>Hello World</h1><p>This is a test page about artificial intelligence.</p></body></html>"

    file_paths = ["/tmp/tos_mount/sample.warc.gz"]
    binary_data = [sample_warc_content]
    base64_data = [f"data:application/octet-stream;base64,{base64.b64encode(sample_warc_content).decode()}"]

    extractor_type = "trafilatura"
    max_records = 5

    df1 = daft.from_pydict({"warc_files": file_paths})
    df1 = df1.with_column(
        "extracted_content",
        las_udf(
            CommonCrawlContentExtractor,
            construct_args={
                "extractor_type": extractor_type,
                "max_records": max_records,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("warc_files")),
    )

    df2 = daft.from_pydict({"warc_files": binary_data})
    df2 = df2.with_column(
        "extracted_content",
        las_udf(
            CommonCrawlContentExtractor,
            construct_args={
                "extractor_type": extractor_type,
                "max_records": max_records,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("warc_files")),
    )

    df3 = daft.from_pydict({"warc_files": base64_data})
    df3 = df3.with_column(
        "extracted_content",
        las_udf(
            CommonCrawlContentExtractor,
            construct_args={
                "extractor_type": extractor_type,
                "max_records": max_records,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("warc_files")),
    )

    print("=== 文件路径输入 ===")
    df1.show()
    print("\n=== 二进制数据输入 ===")
    df2.show()
    print("\n=== Base64编码输入 ===")
    df3.show()

    # ╭──────────────────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ warc_files                                    ┆ extracted_content                                           │
    # │ ---                                          ┆ ---                                                         │
    # │ Utf8                                         ┆ List[Struct[url: Utf8, content: Utf8, warc_file: Utf8,      │
    # │                                              ┆ extractor: Utf8]]                                           │
    # ╞══════════════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ /tmp/tos_mount/sample.warc.gz                ┆ [{url: http://00852imports.com/detail/5389084.html,        │
    # │                                              ┆ content: 随着互联网的发展，人们对网络速度的要求也越来越高…, │
    # │                                              ┆ warc_file: sample.warc.gz, extractor: trafilatura},          │
    # │                                              ┆ {url: http://02y3tcpv.gd9.cc/?penglaibexdkcl224396.html,    │
    # │                                              ┆ content: 查看更多相关内容\n\n取消关注在如今的数字时代…,      │
    # │                                              ┆ warc_file: sample.warc.gz, extractor: trafilatura}]          │
    # ╰──────────────────────────────────────────────┴─────────────────────────────────────────────────────────────╯
    #
    # ╭──────────────────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ warc_files                                    ┆ extracted_content                                           │
    # │ ---                                          ┆ ---                                                         │
    # │ Binary                                       ┆ List[Struct[url: Utf8, content: Utf8, warc_file: Utf8,      │
    # │                                              ┆ extractor: Utf8]]                                           │
    # ╞══════════════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ [binary data]                                ┆ [{url: http://example.com,                                 │
    # │                                              ┆ content: Hello World\nThis is a test page about artificial…, │
    # │                                              ┆ warc_file: [binary_input], extractor: trafilatura}]          │
    # ╰──────────────────────────────────────────────┴─────────────────────────────────────────────────────────────╯
    #
    # ╭──────────────────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ warc_files                                    ┆ extracted_content                                           │
    # │ ---                                          ┆ ---                                                         │
    # │ Utf8                                         ┆ List[Struct[url: Utf8, content: Utf8, warc_file: Utf8,      │
    # │                                              ┆ extractor: Utf8]]                                           │
    # ╞══════════════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ data:application/octet-stream;base64,...     ┆ [{url: http://example.com,                                 │
    # │                                              ┆ content: Hello World\nThis is a test page about artificial…, │
    # │                                              ┆ warc_file: [binary_input], extractor: trafilatura}]          │
    # ╰──────────────────────────────────────────────┴─────────────────────────────────────────────────────────────╯
