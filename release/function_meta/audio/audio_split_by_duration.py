from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioSplitByDuration
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "audio_path": [f"tos://{TOS_TEST_DIR}/audio_split_by_duration/sample.mp3"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to split audio by duration
    output_tos_dir = f"tos://{TOS_TEST_DIR}/audio_split_by_duration"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "output_segments_binary": True,
        "segment_duration": 10.0,
        "min_segment_duration": 1.0,
    }

    ds = ds.with_column(
        "split_results",
        las_udf(AudioSplitByDuration, construct_args=constructor_kwargs)(col("audio_path")),
    )

    ds.show()
    # ╭────────────────────────────────┬────────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ split_results                                             │
    # │ ---                            ┆ ---                                                       │
    # │ Utf8                           ┆ Struct[segments: List[Utf8], binaries: List[Binary]]      │
    # ╞════════════════════════════════╪════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_split_… ┆ {segments: [tos://tos_bucket/…                            │
    # ╰────────────────────────────────┴────────────────────────────────────────────────────────────╯
