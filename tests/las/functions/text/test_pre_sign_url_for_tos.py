# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.pre_sign_url_for_tos import PreSignUrlForTos
from daft.las.functions.udf import las_udf


def test_pre_sign_url_for_other_schema(tos_test_data_dir):
    sample = pd.DataFrame(
        {
            "url": [
                "file://image/cat.png",
                "s3://bucket/image/cat.png",
                f"tos://{tos_test_data_dir}/image/cat.png",
                "dir/image/cat.png",
                "https://bucket/image/cat.png",
                "/dir/image/cat.png",
            ]
        }
    )
    df = daft.from_pandas(sample)

    df = df.with_column(
        "signed_url",
        las_udf(
            PreSignUrlForTos,
            construct_args={"expires": 3600},
        )(col("url")),
    )
    res = df.to_pandas()

    assert res["signed_url"][0] is None and res["signed_url"][3] is None and res["signed_url"][5] is None
    assert res["signed_url"][1].startswith("https://") and "image/cat.png" in res["signed_url"][1]
    assert res["signed_url"][2].startswith("https://") and "image/cat.png" in res["signed_url"][2]
    assert res["signed_url"][4] == "https://bucket/image/cat.png"

    assert "Expires=" in res["signed_url"][1]
    assert "Signature=" in res["signed_url"][1]
