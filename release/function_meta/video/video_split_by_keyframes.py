from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSplitByKeyframes

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"video_path": [f"tos://{TOS_TEST_DIR}/video_split_by_keyframes/sample.mp4"]}

    ds = daft.from_pydict(samples)

    splitter = las_udf(
        VideoSplitByKeyframes,
        construct_args={
            "method": "I_frame",
            "keyframes_cnt": 2,
            "output_tos_dir": f"tos://{TOS_TEST_DIR}/video_split_by_keyframes",
        },
    )

    ds = ds.with_column("results", splitter(col("video_path")))

    ds.show()
    # ╭──────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    # │ video_path                               ┆ results                                                                                                                          │
    # │ ---                                      ┆ ---                                                                                                                              │
    # │ Utf8                                     ┆ Struct[segments: List[Utf8], segments_binary: List[Binary], video_format: List[Utf8]]                                            │
    # ╞══════════════════════════════════════════╪══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/video_split_by_keyfram… ┆ {segments: ["tos://tos_bucket/video_split_by_keyframes/sample/segment_1.mp4", "tos://tos_bucket/video_split_by_keyframes/sam…    │
    # ╰──────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
