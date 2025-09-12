# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.doc import PDFParse
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    return pd.DataFrame(
        {
            "input_url": [
                "",
                f"{local_test_data_dir}/doc/sample.pdf",
                f"{tos_test_data_dir}/doc/sample.pdf",
                f"{http_test_data_dir}/doc/invalid.pdf",
                f"{http_test_data_dir}/doc/sample.pdf",
            ],
            "filename": ["", "sample.pdf", "sample.pdf", "invalid.pdf", "sample.pdf"],
        }
    )


def test_pdf_parse(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_pd_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    df = daft.from_pandas(input_pd_df)

    constructor_kwargs = {
        "input_type": "url",
        "output_tos_path": f"{tos_test_data_dir}/doc/pdf_parse",
        "qps": 1,
    }

    df = df.with_column(
        "result",
        las_udf(PDFParse, construct_args=constructor_kwargs, concurrency=1)(col("input_url"), col("filename")),
    )

    df = df.with_column("parsed_text", col("result").struct.get("parsed_origin_text"))
    df = df.with_column("plain_text", col("result").struct.get("parsed_plain_text"))
    df = df.with_column("detail", col("result").struct.get("parsed_detail"))
    df = df.with_column("file_path", col("result").struct.get("parsed_file_path"))
    df = df.with_column("parsed_image_filenames", col("result").struct.get("parsed_image_filenames"))

    output_pd_df = df.select(
        "input_url", "filename", "parsed_text", "plain_text", "detail", "file_path", "parsed_image_filenames"
    ).to_pandas()

    expect_columns = [
        "input_url",
        "filename",
        "parsed_text",
        "plain_text",
        "detail",
        "file_path",
        "parsed_image_filenames",
    ]
    expect_row_num = len(input_pd_df)

    assert_dataframe_result(
        actual_df=output_pd_df,
        expect_columns=expect_columns,
        expect_row_num=expect_row_num,
    )
