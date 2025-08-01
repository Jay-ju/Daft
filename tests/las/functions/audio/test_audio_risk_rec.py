# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.audio import AudioRiskRec
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(http_test_data_dir):
    paths = [
        "",
        f"{http_test_data_dir}/audio/non-exist.mp3",
        f"{http_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{http_test_data_dir}/audio/sample.mp3",
    ]
    return pd.DataFrame(
        {
            "audio_id": [i + 1 for i in range(len(paths))],
            "audio_path": paths,
        }
    )


def test_audio_risk_rec(http_test_data_dir):
    app_id = os.getenv("APP_ID")
    biztype = os.getenv("BIZTYPE")

    if not app_id or not biztype:
        pytest.skip("APP_ID and BIZTYPE environment variables must be set for this test")

    input_pd_df = generate_test_data(http_test_data_dir)
    df = daft.from_pandas(input_pd_df)

    constructor_kwargs = {"app_id": int(app_id), "biztype": biztype}

    df = df.with_column(
        "result",
        las_udf(AudioRiskRec, construct_args=constructor_kwargs, concurrency=1, batch_size=1)(
            col("audio_id"), col("audio_path")
        ),
    )

    df = df.with_column("Decision", col("result").struct.get("Decision"))
    df = df.with_column("Message", col("result").struct.get("Message"))
    df = df.with_column("risk_result", col("result").struct.get("risk_result"))

    output_pd_df = df.select("audio_id", "audio_path", "Decision", "Message", "risk_result").to_pandas()

    expect_columns = [
        "audio_id",
        "audio_path",
        "Decision",
        "Message",
        "risk_result",
    ]
    expect_row_num = len(input_pd_df)

    assert_dataframe_result(
        actual_df=output_pd_df,
        expect_columns=expect_columns,
        expect_row_num=expect_row_num,
    )
    assert output_pd_df["Decision"].iloc[-1] in ("PASS", "BLOCK", "REVIEW")
