# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioMetascore
from daft.las.functions.udf import las_udf


def test_audio_metascore_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    """Test AudioMetascore operator with default parameters."""
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/audio/sample.wav",
        ],
    }
    input_df = pd.DataFrame(samples)
    constructor_kwargs = {
        "model_path": local_models_dir,
    }

    # 期望结果
    expected_results = [
        {"CE": 5.909880638122559, "CU": 6.179478168487549, "PC": 5.604213237762451, "PQ": 7.392683982849121},
    ]
    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "audio_metascore": expected_results,
        }
    )

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "audio_metascore",
        las_udf(
            AudioMetascore,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path")),
    ).to_pandas()

    import math

    for i, (res, exp) in enumerate(zip(result_df["audio_metascore"], expected_df["audio_metascore"])):
        for key in exp:
            assert key in res, f"Row {i}: Missing key {key}"
            assert math.isclose(
                res[key], exp[key], rel_tol=1e-3, abs_tol=1e-3
            ), f"Row {i}, key {key} differ: {res[key]} vs {exp[key]}"
