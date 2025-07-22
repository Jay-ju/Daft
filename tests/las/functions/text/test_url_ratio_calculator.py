# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.url_ratio_calculator import UrlRatioCalculator
from daft.las.functions.udf import las_udf


def test_url_ratio_calculator_basic():
    test_texts = [
        "Hello world",
        "Check out https://example.com for more info",
        "Visit http://test.com and https://demo.org",
        "https://long-url-with-query-params.com/path?param1=value1&param2=value2#section",
        "No URLs here, just text",
        "Mixed content: https://short.com and some text",
        "Multiple URLs: http://first.com, https://second.org, http://third.net",
        "",
        "     ",
        "https://example.com",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "url_ratio",
        las_udf(
            UrlRatioCalculator,
            construct_args={},
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "url_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["url_ratio"]) == 10

    assert result_list["url_ratio"][0] == 0.0
    assert 0.44 <= result_list["url_ratio"][1] <= 0.45
    assert 0.73 <= result_list["url_ratio"][2] <= 0.74
    assert result_list["url_ratio"][3] > 0.0
    assert result_list["url_ratio"][4] == 0.0
    assert 0.36 <= result_list["url_ratio"][5] <= 0.37
    assert result_list["url_ratio"][6] > 0.0
    assert result_list["url_ratio"][7] is None
    assert result_list["url_ratio"][8] is None
    assert result_list["url_ratio"][9] > 0.0
