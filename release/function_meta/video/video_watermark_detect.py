from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoWatermarkDetect

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "video_path": [f"tos://{TOS_TEST_DIR}/video_watermark_detect/sample.mp4"],
    }
    ds = daft.from_pydict(samples)

    model_path = os.getenv("MODEL_PATH", "./models")
    constructor_kwargs = {
        "model_path": model_path,
        "model_name": "PP-OCRv4/ch_det",
        "sample_count": 15,
        "consistency_threshold": 0.8,
        "position_tolerance": 15,
    }

    ds = ds.with_column(
        "results",
        las_udf(
            VideoWatermarkDetect,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("video_path")),
    )

    ds.show()
# ╭────────────────────────────────┬─────────────────────────────────────────────────────────────────────╮
# │ video_path                     ┆ results                                                             │
# │ ---                            ┆ ---                                                                 │
# │ Utf8                           ┆ Struct[watermark_regions: List[Struct[ymin: Int32, ymax: Int32,     │
# │                                ┆ xmin: Int32, xmax: Int32, confidence: Float32]], video_resolution:  │
# │                                ┆ List[Int32], total_frames: Int64]                                   │
# ╞════════════════════════════════╪═════════════════════════════════════════════════════════════════════╡
# │ tos://tos_bucket/vide…         ┆ {watermark_regions: [{ymin: 8…                                      │
# ╰────────────────────────────────┴─────────────────────────────────────────────────────────────────────╯
