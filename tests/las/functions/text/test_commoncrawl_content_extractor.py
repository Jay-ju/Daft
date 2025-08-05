# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import os
import random

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.text.commoncrawl_content_extractor import CommonCrawlContentExtractor
from daft.las.functions.udf import las_udf

num_gpus = int(os.getenv("NUM_GPUS", 1))
rank = random.randint(0, num_gpus - 1)


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    sample_warc_content = b"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: http://example.com\r\nContent-Type: application/http; msgtype=response\r\nContent-Length: 200\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body><h1>Hello World</h1><p>This is a test page.</p></body></html>"

    return pd.DataFrame(
        {
            "warc_files": [
                "",
                None,
                f"{local_test_data_dir}/warc/non-exist.warc.gz",
                f"{local_test_data_dir}/warc/sample.warc.gz",
                f"{tos_test_data_dir}/warc/sample.warc.gz",
                f"{http_test_data_dir}/warc/sample.warc.gz",
                sample_warc_content,
                f"data:application/octet-stream;base64,{base64.b64encode(sample_warc_content).decode()}",
            ]
        }
    )


@pytest.mark.gpu
@pytest.mark.parametrize("extractor_type", ["justext", "trafilatura", "goose3"])
def test_commoncrawl_content_extractor(tos_test_data_dir, local_test_data_dir, http_test_data_dir, extractor_type):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "extracted_content",
        las_udf(
            CommonCrawlContentExtractor,
            construct_args={
                "extractor_type": extractor_type,
                "max_records": 100,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("warc_files")),
    )

    actual_df = ds.to_pandas()

    assert actual_df["extracted_content"][0] is None or len(actual_df["extracted_content"][0]) == 0
    assert actual_df["extracted_content"][1] is None or len(actual_df["extracted_content"][1]) == 0
    assert actual_df["extracted_content"][2] is None or len(actual_df["extracted_content"][2]) == 0

    for i in [3, 6, 7]:
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
                assert first_record["warc_file"] in ["sample.warc.gz", "[binary_input]"]
                assert first_record["url"].startswith("http") or first_record["url"] == ""

                if first_record["url"]:
                    assert len(first_record["content"].strip()) > 0
                    assert "<" not in first_record["content"] and ">" not in first_record["content"]


def test_invalid_extractor_type():
    with pytest.raises(ValueError, match="Unsupported extractor type"):
        CommonCrawlContentExtractor(extractor_type="invalid_extractor")
