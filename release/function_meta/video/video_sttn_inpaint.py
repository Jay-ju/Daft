from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSttnInpaint

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "video_path": [f"tos://{TOS_TEST_DIR}/video_sttn_inpaint/sample.mp4"],
    }
    ds = daft.from_pydict(samples)

    output_tos_dir = f"tos://{TOS_TEST_DIR}/video_sttn_inpaint"
    model_path = os.getenv("MODEL_PATH", "./models")
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "model_path": model_path,
        "model_name": "researchmm/STTN",
        "neighbor_stride": 5,
        "reference_length": 10,
        "max_load_num": 50,
    }

    ds = ds.with_column(
        "results",
        las_udf(
            VideoSttnInpaint,
            construct_args=constructor_kwargs,
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("video_path")),
    )

    ds.show()
# ╭────────────────────────────────┬────────────────────────────────────────────────────╮
# │ video_path                     ┆ results                                            │
# │ ---                            ┆ ---                                                │
# │ Utf8                           ┆ Struct[output_path: Utf8, processed_frames: Int32, │
# │                                ┆ processed_resolution: List[Int32]]                 │
# ╞════════════════════════════════╪════════════════════════════════════════════════════╡
# │ tos://tos_bucket/vide…         ┆ {output_path: tos://tos_bucket/…                   │
# ╰────────────────────────────────┴────────────────────────────────────────────────────╯
