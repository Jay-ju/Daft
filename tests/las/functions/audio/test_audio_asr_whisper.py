# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import random

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.audio.audio_asr_whisper import AudioAsrWhisper
from daft.las.functions.udf import las_udf

audio_src_type = "audio_url"
dtype = "bfloat16"
source_language = "chinese"
translate_to_english = False
condition_on_prev_tokens = True
compression_ratio_threshold = 1.35
temperature = 0.5
logprob_threshold = -1.0
batch_size = 1
num_gpus = int(os.getenv("NUM_GPUS", 1))
num_gpus = 1
rank = random.randint(0, num_gpus - 1)


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{local_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{tos_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{http_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
    ]
    return pd.DataFrame({"audio_path": paths})


@pytest.mark.gpu
def test_audio_asr_whisper(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrWhisper,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": local_models_dir,
                "model_name": "openai/whisper-large-v3",
                "dtype": dtype,
                "source_language": source_language,
                "translate_to_english": translate_to_english,
                "condition_on_prev_tokens": condition_on_prev_tokens,
                "compression_ratio_threshold": compression_ratio_threshold,
                "temperature": temperature,
                "logprob_threshold": logprob_threshold,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()
    assert (
        actual_df["asr_result_detail"][0]["asr_result"] is None
        or len(actual_df["asr_result_detail"][0]["asr_result"]) == 0
    )
    assert (
        actual_df["asr_result_detail"][1]["asr_result"] is None
        or len(actual_df["asr_result_detail"][1]["asr_result"]) == 0
    )
    assert "浙江省" in actual_df["asr_result_detail"][2]["asr_result"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ([(None, None)], [[0, 0]]),
        ([(None, 5)], [[0, 5]]),
        ([(5, None)], [[5, 0]]),
        ([(0, 4.3467)], [[0, 4.347]]),
        ([(0, 4.34), (4.34, 7.14)], [[0, 4.34], [4.34, 7.14]]),
        ([(0, 4.34), (30, None)], [[0, 4.34], [4.34, 4.34]]),
        ([(0, 4.34), (None, 30)], [[0, 4.34], [4.34, 4.34]]),
        ([(0, 4.34), (4.34, None)], [[0, 4.34], [4.34, 8.68]]),
        ([(0, 4.34), (2, None)], [[0, 4.34], [4.34, 6.34]]),
        ([(0, 4.34), (None, 29)], [[0, 4.34], [4.34, 33.34]]),
        ([(0, 4.34), (None, None)], [[0, 4.34], [4.34, 4.34]]),
        ([(0, 4.34), (4.34, 7.14), (7.14, 10.14)], [[0, 4.34], [4.34, 7.14], [7.14, 10.14]]),
        ([(0, 4.34), (7.14, 5), (7.14, 10.14)], [[0, 4.34], [7.14, 12.14], [14.28, 17.28]]),
        ([(0, 4.34), (3.34, 7.14), (7.14, 10.14)], [[0, 4.34], [30, 37.14], [37.14, 40.14]]),
        (
            [(0, 4.34), (4.34, 7.14), (5.14, 10.14), (10.14, 12.14)],
            [[0, 4.34], [4.34, 7.14], [7.14, 17.28], [17.28, 19.28]],
        ),
        (
            [(0, 4.34), (4.34, 7.14), (5.14, 10.14), (30, None)],
            [[0, 4.34], [4.34, 7.14], [7.14, 17.28], [17.28, 17.28]],
        ),
        (
            [(0, 4.34), (4.34, 7.14), (5.14, 10.14), (None, 30)],
            [[0, 4.34], [4.34, 7.14], [7.14, 17.28], [17.28, 17.28]],
        ),
        (
            [(0, 4.34), (4.34, 7.14), (5.14, 10.14), (10.14, None)],
            [[0, 4.34], [4.34, 7.14], [7.14, 17.28], [17.28, 27.42]],
        ),
        (
            [(0, 4.34), (4.34, 7.14), (5.14, 10.14), (8, None)],
            [[0, 4.34], [4.34, 7.14], [7.14, 17.28], [17.28, 25.28]],
        ),
        (
            [(0, 4.34), (4.34, 7.14), (5.14, 10.14), (None, 29)],
            [[0, 4.34], [4.34, 7.14], [7.14, 17.28], [17.28, 46.28]],
        ),
        (
            [(0, 4.34), (4.34, 7.14), (5.14, 10.14), (None, None)],
            [[0, 4.34], [4.34, 7.14], [7.14, 17.28], [17.28, 17.28]],
        ),
    ],
)
def test_update_timestamps(source, expected):
    actual = AudioAsrWhisper._update_timestamps(source)
    assert actual == expected
