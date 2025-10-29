from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioConvert
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [f"tos://{TOS_TEST_DIR}/audio_convert/sample.wav"],
        "output_path": [f"tos://{TOS_TEST_DIR}/audio_convert/sample_converted.mp3"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to convert audio to MP3 format (as an example)
    # AudioConvert supports multiple formats: mp3, wav, flac, aac, ogg, etc.
    constructor_kwargs = {
        "output_format": "mp3",
        "sample_rate": 44100,
        "audio_map": "auto",
        "extra_params": ["-c:a", "libmp3lame", "-b:a", "192k", "-q:a", "2"],
    }

    ds = ds.with_column(
        "convert_result",
        las_udf(AudioConvert, construct_args=constructor_kwargs)(col("input_path"), col("output_path")),
    )

    ds.show()
    # ╭─────────────────────────────────────────────────┬──────────────────────────────────────────────────┬──────────────────────────────────────────────────╮
    # │ input_path                                      ┆ output_path                                      ┆ convert_result                                   │
    # │ ---                                             ┆ ---                                              ┆ ---                                              │
    # │ Utf8                                            ┆ Utf8                                             ┆ Utf8                                             │
    # ╞═════════════════════════════════════════════════╪══════════════════════════════════════════════════╪══════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_convert/sample.wav       ┆ tos://tos_bucket/audio_convert/sample_convert…   ┆ tos://tos_bucket/audio_convert/sample_convert…   │
    # ╰─────────────────────────────────────────────────┴──────────────────────────────────────────────────┴──────────────────────────────────────────────────╯
