# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.doc import XlsxParse
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/doc/non-exist.xlsx",
        f"{local_test_data_dir}/doc/sample.xlsx",
        f"{tos_test_data_dir}/doc/sample.xlsx",
    ]

    return pd.DataFrame(
        {
            "xlsx_path": paths,
        }
    )


def test_xlsx_parse(tos_test_data_dir, local_test_data_dir):
    input_pd_df = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pandas(input_pd_df)

    constructor_kwargs = {
        "if_save_md_content": True,
        "if_save_html_content": False,
        "output_tos_path": f"{tos_test_data_dir}/doc/xlsx_parse",
    }

    df = df.with_column(
        "result",
        las_udf(XlsxParse, construct_args=constructor_kwargs, concurrency=1)(col("xlsx_path")),
    )

    df = df.with_column("data_item_uri", col("result").struct.get("data_item_uri"))
    df = df.with_column("text", col("result").struct.get("text"))
    df = df.with_column("text_by_table", col("result").struct.get("text_by_table"))

    output_pd_df = df.select("xlsx_path", "data_item_uri", "text", "text_by_table").to_pandas()

    expect_columns = ["xlsx_path", "data_item_uri", "text", "text_by_table"]
    expect_row_num = len(input_pd_df)

    assert_dataframe_result(
        actual_df=output_pd_df,
        expect_columns=expect_columns,
        expect_row_num=expect_row_num,
    )
    assert output_pd_df["text"][3].startswith("|   产品ID | 产品名称")
