from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioMetadataExtract
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [
            f"tos://{TOS_TEST_DIR}/audio_metadata_extract/sample.wav",
        ],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to extract audio metadata
    constructor_kwargs = {
        "timeout": 600,
    }

    ds = ds.with_column(
        "metadata",
        las_udf(AudioMetadataExtract, construct_args=constructor_kwargs)(col("input_path")),
    )

    ds.show()
    # ╭───────────────────────────────────────────────────────────────╮──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    # │ input_path                                                    ┆ metadata                                                                                                         │
    # │ ---                                                           ┆ ---                                                                                                              │
    # │ Utf8                                                          ┆ Struct                                                                                                           │
    # ╞═══════════════════════════════════════════════════════════════╪══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_metadata_extract/sample.wav            ┆ {duration: 49.71102, format_name: "wav", bit_rate: 1411212, audio_codec: "pcm_s16le", audio_sample_rate: 4... │
    # ╰───────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
