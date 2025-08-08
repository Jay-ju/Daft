# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import random

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.commoncrawl_content_extractor import CommonCrawlContentExtractor
from daft.las.functions.udf import las_udf

warc_src_type = "warc_url"
extractor_type = "trafilatura"
max_records = 100
num_gpus = int(os.getenv("NUM_GPUS", 1))
rank = random.randint(0, num_gpus - 1)


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    warc_paths = [
        "",
        f"{local_test_data_dir}/warc/non-exist.warc.gz",
        f"{local_test_data_dir}/warc/sample.warc.gz",
        f"{tos_test_data_dir}/warc/sample.warc.gz",
        f"{http_test_data_dir}/warc/sample.warc.gz",
    ]
    return pd.DataFrame({"warc_path": warc_paths})


def test_commoncrawl_content_extractor(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "extracted_content",
        las_udf(
            CommonCrawlContentExtractor,
            construct_args={
                "warc_src_type": warc_src_type,
                "extractor_type": extractor_type,
                "max_records": max_records,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("warc_path")),
    )

    actual_df = ds.to_pandas()

    assert actual_df["extracted_content"][0] is None or len(actual_df["extracted_content"][0]) == 0
    assert actual_df["extracted_content"][1] is None or len(actual_df["extracted_content"][1]) == 0

    for i in [2, 3, 4]:
        if actual_df["extracted_content"][i] is None:
            continue
        if len(actual_df["extracted_content"][i]) > 0:
            content_list = actual_df["extracted_content"][i].tolist()
            assert isinstance(content_list, list)

            if len(content_list) > 0:
                first_record = content_list[0]

                assert "url" in first_record
                assert "content" in first_record
                assert "warc_file" in first_record
                assert "extractor" in first_record

                assert isinstance(first_record["url"], str)
                assert isinstance(first_record["content"], str)
                assert isinstance(first_record["warc_file"], str)
                assert isinstance(first_record["extractor"], str)

                assert first_record["extractor"] == extractor_type
                assert first_record["url"].startswith("http") or first_record["url"] == ""

                if first_record["url"]:
                    assert len(first_record["content"].strip()) > 0
                    assert "<" not in first_record["content"] and ">" not in first_record["content"]
