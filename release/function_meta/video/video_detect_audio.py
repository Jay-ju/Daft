from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoDetectAudio

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [
            f"tos://{TOS_TEST_DIR}/video_detect_audio/music_sample.mp4",
            f"tos://{TOS_TEST_DIR}/video_detect_audio/music_sample_no_audio.mp4",
        ],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to detect audio in video
    constructor_kwargs = {
        "timeout": 600,
    }

    ds = ds.with_column(
        "has_audio",
        las_udf(VideoDetectAudio, construct_args=constructor_kwargs)(col("input_path")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────────────────────╮──────────────╮
    # │ input_path                                                   ┆ has_audio    │
    # │ ---                                                          ┆ ---          │
    # │ Utf8                                                         ┆ Boolean      │
    # ╞══════════════════════════════════════════════════════════════╪══════════════╡
    # │ tos://tos_bucket/video_detect_audio/sample_with_audio.mp4   ┆ true         │
    # │ tos://tos_bucket/video_detect_audio/sample_without_audio... ┆ false        │
    # ╰──────────────────────────────────────────────────────────────┴──────────────╯
