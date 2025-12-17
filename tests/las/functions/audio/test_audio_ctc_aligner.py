# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import json

import pytest

import daft
from daft import col
from daft.las.functions import las_udf
from daft.las.functions.audio.audio_ctc_aligner import AudioCTCAligner


@pytest.mark.gpu
def test_audio_ctc_alignment(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    audio_ids = ["en", "zh"]
    df = daft.from_pydict(
        {
            "audio_id": audio_ids,
            "audio_path": [f"{tos_test_data_dir}/audio/audio_ctc/{id}.wav" for id in audio_ids],
            "text": ["i had that curiosity beside me at this moment", "将于十一月二十一日早七点在东方明珠塔下跑起"],
            "lang": ["en", "zh"],
        }
    )

    result = (
        df.with_column(
            "ctc_result",
            las_udf(
                AudioCTCAligner,
                construct_args={ "model_path": local_models_dir },
                num_gpus=0.5,
                batch_size=2,
                concurrency=2,
            )(col("audio_path"), col("text"), col("lang")),
        )
        .select(col("audio_id"), col("ctc_result"))
        .to_pydict()
    )

    en_expected_result = [
        {"word": "i", "score": 1.0, "start": 644, "end": 664},
        {"word": "had", "score": 0.98, "start": 704, "end": 845},
        {"word": "that", "score": 1.0, "start": 885, "end": 1026},
        {"word": "curiosity", "score": 1.0, "start": 1086, "end": 1790},
        {"word": "beside", "score": 0.97, "start": 1871, "end": 2314},
        {"word": "me", "score": 1.0, "start": 2334, "end": 2414},
        {"word": "at", "score": 1.0, "start": 2495, "end": 2575},
        {"word": "this", "score": 1.0, "start": 2595, "end": 2756},
        {"word": "moment", "score": 1.0, "start": 2837, "end": 3138},
    ]

    assert en_expected_result == sorted(json.loads(result["ctc_result"][0]), key=lambda x: x["start"])

    zh_expected_result = sorted(
        [
            {"word": "将", "score": 0.61, "start": 501, "end": 822},
            {"word": "于", "score": 0.98, "start": 922, "end": 1042},
            {"word": "十", "score": 0.21, "start": 1303, "end": 1443},
            {"word": "一", "score": 0.01, "start": 1584, "end": 1664},
            {"word": "月", "score": 0.54, "start": 1864, "end": 1944},
            {"word": "二", "score": 0.07, "start": 1964, "end": 2085},
            {"word": "十", "score": 0.01, "start": 2446, "end": 2586},
            {"word": "一", "score": 0.21, "start": 2726, "end": 2806},
            {"word": "日", "score": 0.5, "start": 3027, "end": 3147},
            {"word": "早", "score": 0.96, "start": 3348, "end": 3568},
            {"word": "七", "score": 0.26, "start": 3688, "end": 3909},
            {"word": "点", "score": 0.92, "start": 4049, "end": 4250},
            {"word": "在", "score": 1.0, "start": 4330, "end": 4510},
            {"word": "东", "score": 0.93, "start": 4611, "end": 4811},
            {"word": "方", "score": 0.93, "start": 4851, "end": 5092},
            {"word": "明", "score": 0.86, "start": 5132, "end": 5352},
            {"word": "珠", "score": 0.93, "start": 5392, "end": 5513},
            {"word": "塔", "score": 0.99, "start": 5693, "end": 5853},
            {"word": "下", "score": 1.0, "start": 6054, "end": 6194},
            {"word": "跑", "score": 0.94, "start": 6335, "end": 6535},
            {"word": "起", "score": 0.87, "start": 6615, "end": 6736},
        ],
        key=lambda x: x["start"],
    )

    assert zh_expected_result == sorted(json.loads(result["ctc_result"][1]), key=lambda x: x["start"])
