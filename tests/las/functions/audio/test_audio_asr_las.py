# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.audio.audio_asr_las import LasAsrPoller, LasAsrSubmitter
from daft.las.functions.text import PreSignUrlForTos
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(http_test_data_dir):
    return [
        f"{http_test_data_dir}/audio/non-exist.wav",
        f"{http_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        "",
    ]


@pytest.mark.skipif(not os.getenv("LAS_SERVICE_ENDPOINT", "").startswith("http"),
                    reason="Waiting for set the endpoint of las service.")
def test_audio_asr_las(http_test_data_dir):
    api_key = os.getenv("LAS_API_KEY")
    endpoint = os.getenv("LAS_SERVICE_ENDPOINT")

    paths = generate_test_data(http_test_data_dir)
    input_df = pd.DataFrame({"audio_path": paths})

    df = daft.from_pandas(input_df)

    df = df.with_column("http_url", las_udf(PreSignUrlForTos)(col("audio_path")))

    df = df.with_column(
        "task_id",
        las_udf(
            LasAsrSubmitter,
            construct_args={
                "api_key": api_key,
                "endpoint": endpoint,
                "max_retries": 10,
            },
            num_cpus=1,
            concurrency=1,
            batch_size=3,
        )(col("http_url")),
    ).exclude("http_url")

    df = df.with_column(
        "asr_result",
        las_udf(
            LasAsrPoller,
            construct_args={
                "api_key": api_key,
                "endpoint": endpoint,
            },
            num_cpus=1,
            concurrency=1,
            batch_size=3,
        )(col("audio_path"), col("task_id")),
    ).exclude("task_id")

    df = df.with_columns(
        {
            "asr_result_raw": col("asr_result").struct.get("asr_result_raw"),
            "asr_result_text": col("asr_result").struct.get("asr_result_text"),
            "failed_reason": col("asr_result").struct.get("failed_reason"),
        }
    ).exclude("asr_result", "asr_result_raw")

    expected_df = pd.DataFrame(
        {
            "audio_path": paths,
            "asr_result_text": ["", "全剧大部分都是在中国取景拍摄地点位于浙江省温州市", ""],
            "failed_reason": [
                "[Invalid audio URI] OperatorWrapper Process failed: internal error,audio download failed",
                "",
                "SUBMIT_TASK_FAILED",
            ],
        }
    )

    assert_dataframe_result(
        actual_df=df.to_pandas(),
        expect_df=expected_df,
        expect_columns=["audio_path", "asr_result_text", "failed_reason"],
        expect_row_num=3,
    )
