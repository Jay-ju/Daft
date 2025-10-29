from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoConvertToMp4

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [f"tos://{TOS_TEST_DIR}/video_convert_to_mp4/sample.mp4"],
        "output_path": [f"tos://{TOS_TEST_DIR}/video_convert_to_mp4/sample_converted.mp4"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to convert video to MP4 format
    constructor_kwargs = {
        "video_codec": "libx264",
        "crf": 23,
        "preset": "medium",
        "max_height": 240,
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "select_audio": "auto",
        "extra_params": ["-movflags", "+faststart"],
    }

    ds = ds.with_column(
        "convert_result",
        las_udf(VideoConvertToMp4, construct_args=constructor_kwargs)(col("input_path"), col("output_path")),
    )

    ds.show()
    # ╭────────────────────────────────┬────────────────────────────────┬────────────────────────────────╮
    # │ input_path                     ┆ output_path                    ┆ convert_result                 │
    # │ ---                            ┆ ---                            ┆ ---                            │
    # │ Utf8                           ┆ Utf8                           ┆ Utf8                           │
    # ╞════════════════════════════════╪════════════════════════════════╪════════════════════════════════╡
    # │ tos://las-ai-qa-online/qa/tes… ┆ tos://las-ai-qa-online/qa/tes… ┆ tos://las-ai-qa-online/qa/tes… │
    # ╰────────────────────────────────┴────────────────────────────────┴────────────────────────────────╯
