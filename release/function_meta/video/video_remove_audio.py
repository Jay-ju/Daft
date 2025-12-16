from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoRemoveAudio

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [f"tos://{TOS_TEST_DIR}/video_remove_audio/music_sample.mp4"],
        "output_path": [f"tos://{TOS_TEST_DIR}/video_remove_audio/music_sample_no_audio.mp4"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to remove audio tracks from video
    constructor_kwargs = {
        "output_format": None,
        "extra_params": [],
    }

    ds = ds.with_column(
        "remove_audio_result",
        las_udf(VideoRemoveAudio, construct_args=constructor_kwargs)(col("input_path"), col("output_path")),
    )

    ds.show()
    # ╭─────────────────────────────────────────────────┬──────────────────────────────────────────────────┬──────────────────────────────────────────────────╮
    # │ input_path                                      ┆ output_path                                      ┆ remove_audio_result                              │
    # │ ---                                             ┆ ---                                              ┆ ---                                              │
    # │ Utf8                                            ┆ Utf8                                             ┆ Utf8                                             │
    # ╞═════════════════════════════════════════════════╪══════════════════════════════════════════════════╪══════════════════════════════════════════════════╡
    # │ tos://tos_bucket/video_remove_audio/sample.mp4 ┆ tos://tos_bucket/video_remove_audio/sample_no_… ┆ tos://tos_bucket/video_remove_audio/sample_no_… │
    # ╰─────────────────────────────────────────────────┴──────────────────────────────────────────────────┴──────────────────────────────────────────────────╯
