# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import math

import daft
from daft import col
from daft.las.functions.audio import AudioSNR
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{tos_test_data_dir}/audio/sample.mp3",
    ]
    return {"audio_path": paths}


def test_audio_snr(tos_test_data_dir, local_test_data_dir):
    samples = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pydict(samples)

    snr_udf = las_udf(
        AudioSNR,
        construct_args={
            "n_components": 2,
            "max_iter": 200,
        },
    )

    df = df.with_column("snr", snr_udf(col("audio_path")))
    pd_df = df.to_pandas()

    # Validate structure
    assert_dataframe_result(
        pd_df,
        expect_columns=["audio_path", "snr"],
        expect_row_num=3,
    )

    # Empty path and non-existent local file -> NaN
    assert math.isnan(pd_df.iloc[0]["snr"])
    assert math.isnan(pd_df.iloc[1]["snr"])

    # TOS sample -> finite float
    val = pd_df.iloc[2]["snr"]
    assert isinstance(val, (float,))
    assert math.isfinite(val)
