from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioConcat
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "audio_paths": [
            [
                f"tos://{TOS_TEST_DIR}/audio_concat/sample_a.mp3",
                f"tos://{TOS_TEST_DIR}/audio_concat/sample_b.wav",
            ],
        ],
        "output_path": [f"tos://{TOS_TEST_DIR}/audio_concat/output/concatenated_audio.mp3"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to concatenate audio files
    constructor_kwargs = {
        "output_format": "mp3",
        "sample_rate": 16000,
        "extra_params": ["-b:a", "192k"],
    }

    ds = ds.with_column(
        "output_path",
        las_udf(AudioConcat, construct_args=constructor_kwargs)(col("audio_paths"), col("output_path")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────╮
    # │ audio_paths                                                  ┆ output_path                                              │
    # │ ---                                                          ┆ ---                                                      │
    # │ List[Utf8]                                                   ┆ Utf8                                                     │
    # ╞══════════════════════════════════════════════════════════════╪══════════════════════════════════════════════════════════╡
    # │ [tos://tos_bucket/audio_concat/sample_a.mp3, ...]           ┆ tos://tos_bucket/audio_concat/output/concatenated_au... │
    # ╰──────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────╯
