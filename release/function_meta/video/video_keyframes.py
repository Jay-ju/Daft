from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoKeyframes

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"video_path": [f"tos://{TOS_TEST_DIR}/video_keyframes/sample.mp4"]}

    ds = daft.from_pydict(samples)

    extractor = las_udf(
        VideoKeyframes,
        construct_args={
            "method": "I_frame",
            "keyframes_cnt": 5,
            "output_tos_dir": f"tos://{TOS_TEST_DIR}/video_keyframes",
        },
    )

    ds = ds.with_column("results", extractor(col("video_path")))

    ds.show()
    # ╭────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    # │ video_path                     ┆ results                                                                                                    │
    # │ ---                            ┆ ---                                                                                                        │
    # │ Utf8                           ┆ Struct[keyframes: List[List[List[List[Int64]]]], base64: List[Utf8], timestamps: List[Float64], tos_paths: │
    # │                                ┆ List[Utf8]]                                                                                                │
    # ╞════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/video_keyfra… ┆ {keyframes: [[[[9, 17, 16], […                                                                             │
    # ╰────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
