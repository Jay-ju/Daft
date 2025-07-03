# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoKeyframes


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/video/non-exist.mp4",
        f"{local_test_data_dir}/video/sample.mp4",
        f"{tos_test_data_dir}/video/sample.mp4",
    ]
    return {"video_path": paths}


def test_video_keyframes(tos_test_data_dir, local_test_data_dir):
    input_dict = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pydict(input_dict)

    extractor = las_udf(
        VideoKeyframes,
        construct_args={"method": "I_frame", "output_tos_dir": f"{tos_test_data_dir}/video/video_keyframes"},
    )

    df = df.with_column("results", extractor(col("video_path")))
    df = df.with_column("results.tos_paths", col("results").struct.get("tos_paths"))
    df = df.with_column("results.keyframes", col("results").struct.get("keyframes"))
    actual_df = df.select("video_path", "results.tos_paths", "results.keyframes").to_pandas()

    assert (
        actual_df.iloc[3]["results.tos_paths"][0]
        == f"{tos_test_data_dir}/video/video_keyframes/sample/keyframe_0000.jpg"
    )
    assert actual_df.iloc[3]["results.keyframes"][0][0][0].tolist() == [9, 17, 16]
    assert actual_df.iloc[3]["results.keyframes"][1][0][0].tolist() == [17, 20, 18]
