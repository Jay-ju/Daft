# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import json

import pytest

import daft
from daft import col
from daft.las.functions import las_udf
from daft.las.functions.audio import AudioBeatsClassifier


@pytest.mark.gpu
def test_audio_classify(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    audio_ids = ["DvxsHG1tuo", "0SdAVK79lg", "5xOcMJpTUk"]
    df = daft.from_pydict(
        {
            "audio_id": audio_ids,
            "audio_path": [f"{tos_test_data_dir}/audio/audio_classify/{id}.wav" for id in audio_ids],
        }
    )

    result = (
        df.with_column(
            "classify_result",
            las_udf(
                AudioBeatsClassifier,
                construct_args={
                    "model_path": local_models_dir,
                    "precision": 2,
                },
                num_gpus=0.5,
                batch_size=2,
                concurrency=2,
            )(col("audio_path")),
        )
        .select(col("audio_id"), col("classify_result"))
        .to_pydict()
    )

    actual_results = {}
    for audio_id, classify_result_str in zip(result["audio_id"], result["classify_result"]):
        classify_result = json.loads(classify_result_str)
        sorted_classify_result = sorted(classify_result, key=lambda x: -x["probability"])
        actual_results[audio_id] = sorted_classify_result

    assert set(actual_results.keys()) == set(audio_ids)

    expected_results = {
        "DvxsHG1tuo": [
            {"label": "/m/04rlf", "probability": 0.85},
            {"label": "/m/09x0r", "probability": 0.39},
            {"label": "/m/03qc9zr", "probability": 0.33},
            {"label": "/m/07sr1lc", "probability": 0.27},
            {"label": "/m/07s2xch", "probability": 0.15},
        ],
        "0SdAVK79lg": [
            {"label": "/m/04rlf", "probability": 0.95},
            {"label": "/m/0342h", "probability": 0.54},
            {"label": "/m/04szw", "probability": 0.48},
            {"label": "/m/0fx80y", "probability": 0.48},
            {"label": "/m/02sgy", "probability": 0.3},
        ],
        "5xOcMJpTUk": [
            {"label": "/m/04rlf", "probability": 0.89},
            {"label": "/m/0342h", "probability": 0.79},
            {"label": "/m/09x0r", "probability": 0.76},
            {"label": "/m/04szw", "probability": 0.75},
            {"label": "/m/0fx80y", "probability": 0.71},
        ],
    }

    for audio_id in audio_ids:
        assert (
            actual_results[audio_id] == expected_results[audio_id]
        ), f"Classify result don't match for audio: {audio_id}"
