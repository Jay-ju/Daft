from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoExtractMetadata

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [
            f"tos://{TOS_TEST_DIR}/video_extract_metadata/music_sample.mp4",
        ],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to extract video metadata
    constructor_kwargs = {
        "timeout": 600,
    }

    ds = ds.with_column(
        "metadata",
        las_udf(VideoExtractMetadata, construct_args=constructor_kwargs)(col("input_path")),
    )

    ds.show()
    # ╭─────────────────────────────────────────────────────────────╮─────────────────────────────────────────────────────╮
    # │ input_path                                                  ┆ metadata                                            │
    # │ ---                                                         ┆ ---                                                 │
    # │ Utf8                                                        ┆ Struct                                              │
    # ╞═════════════════════════════════════════════════════════════╪═════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/video_extract_metadata/music_sample.mp4    ┆ {duration: 7.367, format_name: "mov,mp4,m4a,3gp... │
    # ╰─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────╯
