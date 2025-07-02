from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioSplitByTimestamps
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "audio_path": [f"tos://{TOS_TEST_DIR}/audio_split_by_timestamps/sample.mp3"],
        "timestamps": [[(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to split audio by timestamps
    output_tos_dir = f"tos://{TOS_TEST_DIR}/audio_split_by_timestamps"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "output_segments_binary": True,
        "output_audio_format": True,
    }

    ds = ds.with_column(
        "split_results",
        las_udf(AudioSplitByTimestamps, construct_args=constructor_kwargs)(col("timestamps"), col("audio_path")),
    )

    ds.show()
    # ╭────────────────────────────────┬─────────────────────────────┬───────────────────────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ timestamps                  ┆ split_results                                                             │
    # │ ---                            ┆ ---                         ┆ ---                                                                       │
    # │ Utf8                           ┆ List[List[Float64]]         ┆ Struct[segments: List[Utf8], binaries: List[Binary], formats: List[Utf8]] │
    # ╞════════════════════════════════╪═════════════════════════════╪═══════════════════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_split_… ┆ [[0, 5], [5, 10], [10, 15]] ┆ {segments: [tos://tos_bucket/…                                            │
    # ╰────────────────────────────────┴─────────────────────────────┴───────────────────────────────────────────────────────────────────────────╯
