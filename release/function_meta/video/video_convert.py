from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoConvert

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [f"tos://{TOS_TEST_DIR}/video_convert/music_sample.mov"],
        "output_path": [f"tos://{TOS_TEST_DIR}/video_convert/music_sample.mp4"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to convert video format
    constructor_kwargs = {
        "output_format": "mp4",
        "extra_params": ["-crf", "23", "-preset", "medium"],
    }

    ds = ds.with_column(
        "convert_result",
        las_udf(VideoConvert, construct_args=constructor_kwargs)(col("input_path"), col("output_path")),
    )

    ds.show()
    # ╭────────────────────────────────┬────────────────────────────────┬────────────────────────────────╮
    # │ input_path                     ┆ output_path                    ┆ convert_result                 │
    # │ ---                            ┆ ---                            ┆ ---                            │
    # │ Utf8                           ┆ Utf8                           ┆ Utf8                           │
    # ╞════════════════════════════════╪════════════════════════════════╪════════════════════════════════╡
    # │ tos://tos_bucket/video_conve... ┆ tos://tos_bucket/video_conve... ┆ tos://tos_bucket/video_conve... │
    # ╰────────────────────────────────┴────────────────────────────────┴────────────────────────────────╯
