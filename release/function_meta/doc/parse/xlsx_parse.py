from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.doc import XlsxParse
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "xlsx_path": [f"tos://{TOS_TEST_DIR}/xlsx_parse/sample.xlsx"],
    }
    df = daft.from_pydict(samples)

    constructor_kwargs = {
        "if_save_md_content": True,
        "if_save_html_content": False,
        "output_tos_path": f"tos://{TOS_TEST_DIR}/xlsx_parse/output",
    }

    df = df.with_column(
        "result",
        las_udf(XlsxParse, construct_args=constructor_kwargs, concurrency=1)(col("xlsx_path")),
    )
    df = df.with_column("data_item_uri", col("result").struct.get("data_item_uri"))
    df = df.with_column("text", col("result").struct.get("text"))
    df = df.with_column("text_by_table", col("result").struct.get("text_by_table"))

    df.show()
    # ╭────────────────────────────────┬────────────────────────────────────────────────────────────────────┬────────────────────────────────┬────────────────────────────────────────┬────────────────────────────────────────╮
    # │ xlsx_path                      ┆ result                                                             ┆ data_item_uri                  ┆ text                                   ┆ text_by_table                          │
    # │ ---                            ┆ ---                                                                ┆ ---                            ┆ ---                                    ┆ ---                                    │
    # │ Utf8                           ┆ Struct[data_item_uri: Utf8, text: Utf8, text_by_table: List[Utf8]] ┆ Utf8                           ┆ Utf8                                   ┆ List[Utf8]                             │
    # ╞════════════════════════════════╪════════════════════════════════════════════════════════════════════╪════════════════════════════════╪════════════════════════════════════════╪════════════════════════════════════════╡
    # │ tos://tos_bucket/xlsx_parse…   ┆ {data_item_uri: tos://tos_bucket.…                                 ┆ tos://tos_bucket…              ┆ |   产品ID | 产品名称   |   价格 |   …    ┆ [|   产品ID | 产品名称   |   价格 |  …    │
    # ╰────────────────────────────────┴────────────────────────────────────────────────────────────────────┴────────────────────────────────┴────────────────────────────────────────┴────────────────────────────────────────╯
