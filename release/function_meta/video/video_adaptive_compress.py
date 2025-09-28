from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoAdaptiveCompress

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "video_path": [f"tos://{TOS_TEST_DIR}/video_adaptive_compress/sample.mp4"],
    }
    ds = daft.from_pydict(samples)

    output_tos_dir = f"tos://{TOS_TEST_DIR}/video_adaptive_compress"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "max_output_size_mb": 50.0,
        "target_fps": 5.0,
        "min_resolution_height": 360,
        "rank": None,
    }

    ds = ds.with_column(
        "compressed_video_path",
        las_udf(VideoAdaptiveCompress, construct_args=constructor_kwargs, num_gpus=1, batch_size=1, concurrency=1)(
            col("video_path")
        ),
    )

    ds.show()
    # ╭────────────────────────────────┬──────────────────────────────────╮
    # │ video_path                     ┆ compressed_video_path            │
    # │ ---                            ┆ ---                              │
    # │ Utf8                           ┆ Utf8                             │
    # ╞════════════════════════════════╪══════════════════════════════════╡
    # │ tos://tos_bucket/video_adapt…  ┆ tos://tos_bucket/video_adapt…    │
    # ╰────────────────────────────────┴──────────────────────────────────╯
