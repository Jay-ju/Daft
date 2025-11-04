# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.multimodal.qwen_omni_audio_understanding import QwenOmniAudioUnderstanding
from daft.las.functions.udf import las_udf

daft.set_execution_config(actor_udf_ready_timeout=600)

model_name = "Qwen/Qwen2.5-Omni-7B"

dtype = "bfloat16"
use_flash_attention_2 = True
prompt = "请给出这个音频的详细描述。"
max_caption_length = 256
batch_size = 1
rank = None
num_gpus = 1


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = ["", f"{local_test_data_dir}/audio/non-exist.wav", f"{local_test_data_dir}/audio/sample.mp3"]
    return pd.DataFrame({"audio_path": paths})


def test_qwen_omni_audio_understanding(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenOmniAudioUnderstanding,
            construct_args={
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "use_flash_attention_2": use_flash_attention_2,
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()

    assert actual_df["caption"][0] is None or len(actual_df["caption"][0]) == 0
    assert actual_df["caption"][1] is None or len(actual_df["caption"][1]) == 0
    assert "老孙" in actual_df["caption"][2]
