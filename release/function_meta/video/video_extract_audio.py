from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoExtractAudio

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "video_path": [f"tos://{TOS_TEST_DIR}/video_extract_audio/sample.mp4"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to extract audio from video
    output_tos_dir = f"tos://{TOS_TEST_DIR}/video_extract_audio"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "output_audio_binary": True,
        "output_sampling_rate": 16000,
        "output_audio_format": "mp3",
    }

    ds = ds.with_column(
        "extract_results",
        las_udf(VideoExtractAudio, construct_args=constructor_kwargs)(col("video_path")),
    )

    ds.show()
    # ╭────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    # │ video_path                     ┆ extract_results                                                                                                                          │
    # │ ---                            ┆ ---                                                                                                                                      │
    # │ Utf8                           ┆ Struct[audio_paths: List[Utf8], audio_arrays: List[List[Float32]], binaries: List[Binary], original_audio_sampling_rates: List[Float64]] │
    # ╞════════════════════════════════╪══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/video_extrac… ┆ {audio_paths: [tos://las-ai-c…                                                                                                           │
    # ╰────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
