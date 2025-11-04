# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import math
import os
import random

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_speaker_verification_eres2net import AudioSpeakerVerificationEres2net
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.audio_utils import decode_audio, encode_audio

audio_src_type = "audio_url"
num_gpus = int(os.getenv("NUM_GPUS", 1))
num_gpus = 1
rank = random.randint(0, num_gpus - 1)


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    speaker_a_paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{local_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{http_test_data_dir}/audio/speaker1_a_cn_16k.wav",
        f"{tos_test_data_dir}/audio/speaker1_a_cn_16k.wav",
    ]
    speaker_b_paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{local_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{http_test_data_dir}/audio/speaker1_b_cn_16k.wav",
        f"{tos_test_data_dir}/audio/speaker2_a_cn_16k.wav",
    ]
    return pd.DataFrame({"speaker_a": speaker_a_paths, "speaker_b": speaker_b_paths})


def generate_test_data_base64(tos_test_data_dir, http_test_data_dir):
    speaker_a_paths = [
        f"{http_test_data_dir}/audio/speaker1_a_cn_16k.wav",
        f"{tos_test_data_dir}/audio/speaker1_a_cn_16k.wav",
    ]
    speaker_b_paths = [
        f"{http_test_data_dir}/audio/speaker1_b_cn_16k.wav",
        f"{tos_test_data_dir}/audio/speaker2_a_cn_16k.wav",
    ]
    speaker_a_audio = [decode_audio(audio, 16000, 1) for audio in speaker_a_paths]
    speaker_a_audio_base64 = [encode_audio(audio, "WAV", True) for audio in speaker_a_audio]
    speaker_b_audio = [decode_audio(audio, 16000, 1) for audio in speaker_b_paths]
    speaker_b_audio_base64 = [encode_audio(audio, "WAV", True) for audio in speaker_b_audio]

    return pd.DataFrame(
        {"speaker_a_audio_base64": speaker_a_audio_base64, "speaker_b_audio_base64": speaker_b_audio_base64}
    )


def generate_test_data_binary(tos_test_data_dir, http_test_data_dir):
    speaker_a_paths = [
        f"{http_test_data_dir}/audio/speaker1_a_cn_16k.wav",
        f"{tos_test_data_dir}/audio/speaker1_a_cn_16k.wav",
    ]
    speaker_b_paths = [
        f"{http_test_data_dir}/audio/speaker1_b_cn_16k.wav",
        f"{tos_test_data_dir}/audio/speaker2_a_cn_16k.wav",
    ]
    speaker_a_audio = [decode_audio(audio, 16000, 1) for audio in speaker_a_paths]
    speaker_a_audio_binary = [encode_audio(audio, "WAV", False) for audio in speaker_a_audio]
    speaker_b_audio = [decode_audio(audio, 16000, 1) for audio in speaker_b_paths]
    speaker_b_audio_binary = [encode_audio(audio, "WAV", False) for audio in speaker_b_audio]
    return pd.DataFrame(
        {"speaker_a_audio_binary": speaker_a_audio_binary, "speaker_b_audio_binary": speaker_b_audio_binary}
    )


def test_audio_speaker_verification_eres2net(
    local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir
):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "speaker_verification_result",
        las_udf(
            AudioSpeakerVerificationEres2net,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": local_models_dir,
                "model_name": "iic/speech_eres2net_sv_zh-cn_16k-common",
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("speaker_a"), col("speaker_b")),
    )

    actual_df = ds.to_pandas()
    print(actual_df)

    assert math.isnan(actual_df["speaker_verification_result"][0])
    assert math.isnan(actual_df["speaker_verification_result"][1])
    assert actual_df["speaker_verification_result"][2] == 1
    assert math.isclose(actual_df["speaker_verification_result"][4], 0.71265, rel_tol=1e-3)
    assert math.isclose(actual_df["speaker_verification_result"][5], 0.02866, rel_tol=2e-3)


def test_audio_speaker_verification_eres2net_base64(
    local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir
):
    input_df = generate_test_data_base64(tos_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "speaker_verification_result",
        las_udf(
            AudioSpeakerVerificationEres2net,
            construct_args={
                "audio_src_type": "audio_base64",
                "model_path": local_models_dir,
                "model_name": "iic/speech_eres2net_sv_zh-cn_16k-common",
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("speaker_a_audio_base64"), col("speaker_b_audio_base64")),
    )

    actual_df = ds.to_pandas()
    print(actual_df)
    assert math.isclose(actual_df["speaker_verification_result"][0], 0.71265, rel_tol=1e-3)
    assert math.isclose(actual_df["speaker_verification_result"][1], 0.02866, rel_tol=2e-3)


def test_audio_speaker_verification_eres2net_binary(
    local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir
):
    input_df = generate_test_data_binary(tos_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "speaker_verification_result",
        las_udf(
            AudioSpeakerVerificationEres2net,
            construct_args={
                "audio_src_type": "audio_binary",
                "model_path": local_models_dir,
                "model_name": "iic/speech_eres2net_sv_zh-cn_16k-common",
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("speaker_a_audio_binary"), col("speaker_b_audio_binary")),
    )

    actual_df = ds.to_pandas()
    print(actual_df)
    assert math.isclose(actual_df["speaker_verification_result"][0], 0.71265, rel_tol=1e-3)
    assert math.isclose(actual_df["speaker_verification_result"][1], 0.02866, rel_tol=2e-3)
