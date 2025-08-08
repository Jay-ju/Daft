from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.commoncrawl_content_extractor import CommonCrawlContentExtractor
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")

    extractor_type = "trafilatura"
    max_records = 5

    file_paths = [f"tos://{TOS_TEST_DIR}/commoncrawl_content_extractor/sample.warc.gz"]
    df = daft.from_pydict({"warc_data": file_paths})
    df = df.with_column(
        "extracted_content",
        las_udf(
            CommonCrawlContentExtractor,
            construct_args={
                "warc_src_type": "warc_url",
                "extractor_type": extractor_type,
                "max_records": max_records,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("warc_data")),
    )
    df.show()

    # ╭──────────────────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ warc_files                                   ┆ extracted_content                                           │
    # │ ---                                          ┆ ---                                                         │
    # │ Utf8                                         ┆ List[Struct[url: Utf8, content: Utf8, warc_file: Utf8,      │
    # │                                              ┆ extractor: Utf8]]                                           │
    # ╞══════════════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ /tmp/tos_mount/sample.warc.gz                ┆ [{url: http://00852imports.com/detail/5389084.html,         │
    # │                                              ┆ content: 随着互联网的发展，人们对网络速度的要求也越来越高…,         │
    # │                                              ┆ warc_file: sample.warc.gz, extractor: trafilatura},         │
    # │                                              ┆ {url: http://02y3tcpv.gd9.cc/?penglaibexdkcl224396.html,    │
    # │                                              ┆ content: 查看更多相关内容\n\n取消关注在如今的数字时代…,            │
    # │                                              ┆ warc_file: sample.warc.gz, extractor: trafilatura}]         │
    # ╰──────────────────────────────────────────────┴─────────────────────────────────────────────────────────────╯
