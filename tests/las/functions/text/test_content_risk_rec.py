# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.text import ContentRiskRec
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data():
    samples = {
        "id": [1, 2],
        "text": [
            "今天天气真好，我们一起去公园玩吧！",  # Safe text
            "出售枪支，联系电话123456789",  # Risky text
        ],
        "account_id": ["user1", "user2"],
        "nickname": ["用户1", "用户2"],
    }
    return pd.DataFrame(samples)


def test_content_risk_rec():
    app_id = os.getenv("APP_ID")
    biztype = os.getenv("BIZTYPE")

    if not app_id or not biztype:
        pytest.skip("APP_ID and BIZTYPE environment variables must be set for this test")

    input_pd_df = generate_test_data()
    df = daft.from_pandas(input_pd_df)

    constructor_kwargs = {"app_id": int(app_id), "biztype": biztype}

    df = df.with_column(
        "result",
        las_udf(ContentRiskRec, construct_args=constructor_kwargs, concurrency=1)(
            col("text"), account_id_col=col("account_id"), nick_name_col=col("nickname")
        ),
    )

    df = df.with_column("FinalLabel", col("result").struct.get("FinalLabel"))
    df = df.with_column("Decision", col("result").struct.get("Decision"))
    df = df.with_column("Message", col("result").struct.get("Message"))
    df = df.with_column("risk_result", col("result").struct.get("risk_result"))

    output_pd_df = df.select(
        "id", "text", "account_id", "nickname", "FinalLabel", "Decision", "Message", "risk_result"
    ).to_pandas()

    expect_columns = [
        "id",
        "text",
        "account_id",
        "nickname",
        "FinalLabel",
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
    # Check PASS/BLOCK for the two samples.
    assert output_pd_df["Decision"][0] == "PASS"
    assert output_pd_df["Decision"][1] == "BLOCK"
