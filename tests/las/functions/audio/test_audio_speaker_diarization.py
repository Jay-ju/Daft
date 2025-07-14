# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import random

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_speaker_diarization import AudioSpeakerDiarization
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result

num_gpus = int(os.getenv("NUM_GPUS", 1))
rank = random.randint(0, num_gpus - 1)


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{local_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{tos_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{http_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
    ]

    input_df = pd.DataFrame({"audio_path": paths})
    expected_df = pd.DataFrame(
        {
            "audio_path": paths,
            "audio_speak_diarize": [
                None,
                None,
                [
                    {"end": 5.448, "speaker": "SPEAKER_01", "start": 0.031},
                    {"end": 1.837, "speaker": "SPEAKER_00", "start": 1.027},
                    {"end": 3.676, "speaker": "SPEAKER_00", "start": 2.579},
                    {"end": 7.068, "speaker": "SPEAKER_00", "start": 5.414},
                    {"end": 12.755, "speaker": "SPEAKER_01", "start": 5.566},
                    {"end": 13.902, "speaker": "SPEAKER_02", "start": 13.042},
                    {"end": 17.969, "speaker": "SPEAKER_01", "start": 13.21},
                ],
                [
                    {"end": 4.368, "speaker": "SPEAKER_00", "start": 0.824},
                    {"end": 7.169, "speaker": "SPEAKER_00", "start": 4.79},
                ],
                [
                    {"end": 4.368, "speaker": "SPEAKER_00", "start": 0.824},
                    {"end": 7.169, "speaker": "SPEAKER_00", "start": 4.79},
                ],
                [
                    {"end": 4.368, "speaker": "SPEAKER_00", "start": 0.824},
                    {"end": 7.169, "speaker": "SPEAKER_00", "start": 4.79},
                ],
            ],
        }
    )

    return input_df, expected_df


def test_diarization_audio_speaker(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "audio_speak_diarize",
        las_udf(
            AudioSpeakerDiarization,
            construct_args={"model_path": local_models_dir, "rank": rank},
            num_gpus=num_gpus,
            concurrency=1,
        )(col("audio_path")),
    )
    actual = ds.to_pandas()
    assert_dataframe_result(actual, expected_df)
