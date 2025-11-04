# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest
import torch

import daft
from daft import col
from daft.las.functions.audio.audio_asr_firered import AudioAsrFireRed
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.common_utils import load_file

daft.set_execution_config(actor_udf_ready_timeout=600)

audio_src_type = "audio_url"
model_name = "FireRedAsr/FireRedASR-AED-L"
beam_size = 3
decode_min_len = 0
decode_max_len = 100
nbest = 1
softmax_smoothing = 1.25
aed_length_penalty = 0.6
eos_penalty = 1.0
repetition_penalty = 3.0
llm_length_penalty = 1.0
temperature = 1.0
use_fp16 = True
batch_size = 1

num_gpus = torch.cuda.device_count()
if num_gpus == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
num_gpus = 1


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{local_test_data_dir}/audio/sample_normal.wav",
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。_normal.wav",
        f"{tos_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。_normal.wav",
        f"{http_test_data_dir}/audio/speaker2_a_cn_16k.wav",
    ]
    return pd.DataFrame({"audio_path": paths})


def generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。_normal.wav",
        f"{tos_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。_normal.wav",
        f"{http_test_data_dir}/audio/speaker2_a_cn_16k.wav",
    ]
    audios_base64 = [load_file(audio, as_base64=True) for audio in paths]
    return pd.DataFrame({"audio_base64": audios_base64})


def generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。_normal.wav",
        f"{tos_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。_normal.wav",
        f"{http_test_data_dir}/audio/speaker2_a_cn_16k.wav",
    ]
    audios_binary = [load_file(audio, as_base64=False) for audio in paths]
    return pd.DataFrame({"audio_binary": audios_binary})


@pytest.mark.gpu
def test_audio_asr_firered(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrFireRed,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "batch_size": batch_size,
                "beam_size": beam_size,
                "decode_min_len": decode_min_len,
                "decode_max_len": decode_max_len,
                "nbest": nbest,
                "softmax_smoothing": softmax_smoothing,
                "aed_length_penalty": aed_length_penalty,
                "eos_penalty": eos_penalty,
                "repetition_penalty": repetition_penalty,
                "llm_length_penalty": llm_length_penalty,
                "use_fp16": use_fp16,
                "temperature": temperature,
            },
            num_gpus=num_gpus,
            batch_size=batch_size,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["asr_result_detail"][0] is None or len(actual_df["asr_result_detail"][0]) == 0
    assert actual_df["asr_result_detail"][1] is None or len(actual_df["asr_result_detail"][1]) == 0
    assert "老孙" in actual_df["asr_result_detail"][2]


@pytest.mark.skip(reason="skip repeated test.")
def test_audio_asr_firered_base64(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrFireRed,
            construct_args={
                "audio_src_type": "audio_base64",
                "model_path": local_models_dir,
                "model_name": "FireRedAsr/FireRedASR-LLM-L",
                "batch_size": batch_size,
                "beam_size": beam_size,
                "decode_min_len": decode_min_len,
                "decode_max_len": decode_max_len,
                "nbest": nbest,
                "softmax_smoothing": softmax_smoothing,
                "aed_length_penalty": aed_length_penalty,
                "eos_penalty": eos_penalty,
                "repetition_penalty": repetition_penalty,
                "llm_length_penalty": llm_length_penalty,
                "use_fp16": use_fp16,
                "temperature": temperature,
            },
            num_gpus=num_gpus,
            batch_size=batch_size,
            concurrency=1,
        )(col("audio_base64")),
    )

    actual_df = ds.to_pandas()
    assert "浙江" in actual_df["asr_result_detail"][0]


@pytest.mark.skip(reason="skip repeated test.")
def test_audio_asr_firered_binary(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrFireRed,
            construct_args={
                "audio_src_type": "audio_binary",
                "model_path": local_models_dir,
                "model_name": model_name,
                "batch_size": batch_size,
                "beam_size": beam_size,
                "decode_min_len": decode_min_len,
                "decode_max_len": decode_max_len,
                "nbest": nbest,
                "softmax_smoothing": softmax_smoothing,
                "aed_length_penalty": aed_length_penalty,
                "eos_penalty": eos_penalty,
                "repetition_penalty": repetition_penalty,
                "llm_length_penalty": llm_length_penalty,
                "use_fp16": use_fp16,
                "temperature": temperature,
            },
            num_gpus=num_gpus,
            batch_size=batch_size,
            concurrency=1,
        )(col("audio_binary")),
    )

    actual_df = ds.to_pandas()
    assert "浙江" in actual_df["asr_result_detail"][0]
