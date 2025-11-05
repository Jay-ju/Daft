from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSplitByTimestamps

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "video_path": [f"tos://{TOS_TEST_DIR}/video_split_by_timestamps/sample.mp4"],
        "timestamp_ranges": [[(0.0, 2.0), (2.0, 4.0)]],
    }

    ds = daft.from_pydict(samples)

    splitter = las_udf(
        VideoSplitByTimestamps,
        construct_args={
            "output_tos_dir": f"tos://{TOS_TEST_DIR}/video_split_by_timestamps",
            "output_segments_binary": False,
            "output_video_format": "mp4",
        },
    )

    ds = ds.with_column("results", splitter(col("video_path"), None, None, col("timestamp_ranges")))

    ds.show()
    # ╭────────────────────────────────┬────────────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ video_path                     ┆ timestamp_ranges                       ┆ results                                                     │
    # │ ---                            ┆ ---                                    ┆ ---                                                         │
    # │ Utf8                           ┆ List[Struct[_0: Float64, _1: Float64]] ┆ Struct[segments: List[Utf8], segments_binary: List[Binary]] │
    # ╞════════════════════════════════╪════════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://las-ai-cn-beijing/qa/op… ┆ [{_0: 0,                               ┆ {segments: [tos://las-ai-cn-b…                              │
    # │                                ┆ _1: 2,                                 ┆                                                             │
    # │                                ┆ }, {_0: 2,                             ┆                                                             │
    # │                                ┆ _1…                                    ┆                                                             │
    # ╰────────────────────────────────┴────────────────────────────────────────┴─────────────────────────────────────────────────────────────╯
