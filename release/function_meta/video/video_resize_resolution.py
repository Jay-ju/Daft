from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoResizeResolution

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "video_path": [f"tos://{TOS_TEST_DIR}/video_resize_resolution/sample.mp4"],
    }
    ds = daft.from_pydict(samples)

    output_tos_dir = f"tos://{TOS_TEST_DIR}/video_resize_resolution"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "min_width": 1280,
        "max_width": 2560,
        "min_height": 720,
        "max_height": 1440,
        "force_original_aspect_ratio_type": "decrease",
        "rank": None,
    }

    ds = ds.with_column(
        "resized_video_path",
        las_udf(VideoResizeResolution, construct_args=constructor_kwargs, num_gpus=1, batch_size=1, concurrency=1)(
            col("video_path")
        ),
    )

    ds.show()
    # ╭────────────────────────────────┬──────────────────────────────────╮
    # │ video_path                     ┆ resized_video_path               │
    # │ ---                            ┆ ---                              │
    # │ Utf8                           ┆ Utf8                             │
    # ╞════════════════════════════════╪══════════════════════════════════╡
    # │ tos://tos_bucket/video_resize… ┆ tos://tos_bucket/video_resize…   │
    # ╰────────────────────────────────┴──────────────────────────────────╯
