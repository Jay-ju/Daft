# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.doc import PDFParse
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    TOS_TEST_DIR_URL = os.getenv("TOS_TEST_DIR_URL", "tos_bucket.tos-cn-beijing.volces.com")
    samples = {
        "input_url": [f"https://{TOS_TEST_DIR_URL}/pdf_parse/sample.pdf"],
        "filename": ["sample.pdf"],
    }
    df = daft.from_pydict(samples)

    constructor_kwargs = {"input_type": "url", "output_tos_path": f"tos://{TOS_TEST_DIR}/pdf_parse", "qps": 1}

    # 使用 Daft 进行分布式处理
    df = df.with_column(
        "parsed_result",
        las_udf(PDFParse, construct_args=constructor_kwargs, concurrency=1)(col("input_url"), col("filename")),
    )
    df = df.with_column("parsed_text", col("parsed_result").struct.get("parsed_origin_text"))

    df.show()

    # ╭────────────────────────────────┬────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────╮
    # │ input_url                      ┆ filename   ┆ parsed_result                                                                                          ┆ parsed_text                    │
    # │ ---                            ┆ ---        ┆ ---                                                                                                    ┆ ---                            │
    # │ Utf8                           ┆ Utf8       ┆ Struct[parsed_origin_text: Utf8, parsed_plain_text: Utf8, parsed_detail: Utf8, parsed_file_path: Utf8, ┆ Utf8                           │
    # │                                ┆            ┆ parsed_image_filenames: List[Utf8]]                                                                    ┆                                │
    # ╞════════════════════════════════╪════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════╪════════════════════════════════╡
    # │ https://tos_bucket/pdf_parse/… ┆ sample.pdf ┆ {parsed_origin_text: ![fig_94…                                                                         ┆ ![fig_94052](https://pdf-buck… │
    # ╰────────────────────────────────┴────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────╯
