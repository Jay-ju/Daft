# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.doc.doc_convert import DocConvert
from daft.las.functions.udf import las_udf


def generate_test_data(tos_test_data_dir, http_test_data_dir, source_fmt="docx"):
    paths = [
        f"{http_test_data_dir}/doc/sample.{source_fmt}",
        f"{tos_test_data_dir}/doc/sample.{source_fmt}",
    ]
    return pd.DataFrame({f"{source_fmt}": paths})


def expected_df(tos_test_data_dir, http_test_data_dir, source_fmt="docx", target_fmt="pdf"):
    docx_path = [
        f"{http_test_data_dir}/doc/sample.{source_fmt}",
        f"{tos_test_data_dir}/doc/sample.{source_fmt}",
    ]
    result_path = [
        f"{tos_test_data_dir}/doc/doc_convert/source_{source_fmt}_format/sample.{target_fmt}",
        f"{tos_test_data_dir}/doc/doc_convert/source_{source_fmt}_format/sample.{target_fmt}",
    ]
    return pd.DataFrame({source_fmt: docx_path, target_fmt: result_path})


@pytest.mark.parametrize("target_format", ["pdf", "odt", "html", "txt", "docx"])
def test_docx_convert(local_models_dir, tos_test_data_dir, http_test_data_dir, target_format):
    input_df = generate_test_data(tos_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        f"{target_format}",
        las_udf(
            DocConvert,
            construct_args={
                "target_format": target_format,
                "output_dir": f"{tos_test_data_dir}/doc/doc_convert/source_docx_format",
            },
            batch_size=2,
            concurrency=1,
        )(col("docx")),
    )

    actual = ds.to_pandas()
    expected = expected_df(tos_test_data_dir, http_test_data_dir, source_fmt="docx", target_fmt=target_format)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("target_format", ["pdf", "odt", "html", "txt", "docx"])
def test_doc_convert(local_models_dir, tos_test_data_dir, http_test_data_dir, target_format):
    input_df = generate_test_data(tos_test_data_dir, http_test_data_dir, source_fmt="doc")

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        f"{target_format}",
        las_udf(
            DocConvert,
            construct_args={
                "target_format": target_format,
                "output_dir": f"{tos_test_data_dir}/doc/doc_convert/source_doc_format",
            },
            batch_size=2,
            concurrency=1,
        )(col("doc")),
    )

    actual = ds.to_pandas()
    expected = expected_df(tos_test_data_dir, http_test_data_dir, source_fmt="doc", target_fmt=target_format)
    pd.testing.assert_frame_equal(actual, expected)
