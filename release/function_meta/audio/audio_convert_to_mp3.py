from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioConvertToMp3
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [f"tos://{TOS_TEST_DIR}/audio_convert_to_mp3/sample.wav"],
        "output_path": [f"tos://{TOS_TEST_DIR}/audio_convert_to_mp3/sample_converted.mp3"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to convert audio to MP3 format
    constructor_kwargs = {
        "bitrate": "192k",
        "sample_rate": 44100,
        "audio_map": "auto",
        "quality": 2,
        "extra_params": [],
    }

    ds = ds.with_column(
        "convert_result",
        las_udf(AudioConvertToMp3, construct_args=constructor_kwargs)(col("input_path"), col("output_path")),
    )

    ds.show()
    # ╭─────────────────────────────────────────────────┬──────────────────────────────────────────────────┬──────────────────────────────────────────────────╮
    # │ input_path                                      ┆ output_path                                      ┆ convert_result                                   │
    # │ ---                                             ┆ ---                                              ┆ ---                                              │
    # │ Utf8                                            ┆ Utf8                                             ┆ Utf8                                             │
    # ╞═════════════════════════════════════════════════╪══════════════════════════════════════════════════╪══════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_convert_to_mp3/sample… ┆ tos://tos_bucket/audio_convert_to_mp3/sample_c… ┆ tos://tos_bucket/audio_convert_to_mp3/sample_c… │
    # ╰─────────────────────────────────────────────────┴──────────────────────────────────────────────────┴──────────────────────────────────────────────────╯
