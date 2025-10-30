# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.audio.audio_ffmpeg_wrapped import AudioFFMPEGWrapped
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{tos_test_data_dir}/audio/sample.mp3",
    ]
    return {"audio_path": paths}


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_audio_path = f"{tos_test_data_dir}/audio/sample.mp3"
    audio_binary = load_file(sample_audio_path)
    output_basename = "my_audio_202408"

    samples = {
        "audio_path": [None],
        "audio_binary": [audio_binary],
        "audio_format": ["mp3"],
        "output_basename": [output_basename],
    }
    return samples


def test_audio_ffmpeg_wrapped_basic(tos_test_data_dir, local_test_data_dir):
    input_dict = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pydict(input_dict)

    op = las_udf(
        AudioFFMPEGWrapped,
        construct_args={
            "filter_name": "volume",
            "filter_kwargs": {"volume": 1.2},
            "global_args": ["-loglevel", "error", "-hide_banner", "-y"],
            "output_audio_binary": True,
            "output_audio_format": "wav",
            "output_tos_dir": "",
        },
    )

    df = df.with_column("results", op(col("audio_path")))
    df = df.select(
        "audio_path",
        col("results").struct.get("processed_audio_path").alias("processed_audio_path"),
        col("results").struct.get("processed_audio_binary").alias("processed_audio_binary"),
    )
    pd_df = df.to_pandas()

    assert_dataframe_result(
        pd_df,
        expect_columns=["audio_path", "processed_audio_path", "processed_audio_binary"],
        expect_row_num=3,
    )

    assert pd_df.iloc[0]["processed_audio_path"] in ("", None)
    assert pd_df.iloc[1]["processed_audio_path"] in ("", None)

    assert isinstance(pd_df.iloc[2]["processed_audio_binary"], (bytes, bytearray))
    assert pd_df.iloc[2]["processed_audio_path"] == "" or isinstance(pd_df.iloc[2]["processed_audio_path"], str)


def test_audio_ffmpeg_wrapped_binary(tos_test_data_dir):
    samples = generate_test_data_binary(tos_test_data_dir)
    df = daft.from_pydict(samples)

    op = las_udf(
        AudioFFMPEGWrapped,
        construct_args={
            "filter_name": "highpass",
            "filter_kwargs": {"f": 300, "width_type": "h", "width": 0.5},
            "global_args": ["-loglevel", "error", "-hide_banner", "-y"],
            "output_audio_binary": True,
            "output_audio_format": "wav",
            "output_tos_dir": "",
        },
    )

    df = df.with_column(
        "results",
        op(
            col("audio_path"),
            col("audio_binary"),
            col("audio_format"),
            col("output_basename"),
        ),
    )
    df = df.select(
        "audio_path",
        "audio_binary",
        "audio_format",
        "output_basename",
        col("results").struct.get("processed_audio_path").alias("processed_audio_path"),
        col("results").struct.get("processed_audio_binary").alias("processed_audio_binary"),
    )
    pd_df = df.to_pandas()

    assert_dataframe_result(
        pd_df,
        expect_columns=[
            "audio_path",
            "audio_binary",
            "audio_format",
            "output_basename",
            "processed_audio_path",
            "processed_audio_binary",
        ],
        expect_row_num=1,
    )
    assert isinstance(pd_df.iloc[0]["processed_audio_binary"], (bytes, bytearray))
    assert pd_df.iloc[0]["processed_audio_path"] == "" or isinstance(pd_df.iloc[0]["processed_audio_path"], str)
