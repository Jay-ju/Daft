from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioConcatFast
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "audio_paths": [
            [
                f"tos://{TOS_TEST_DIR}/audio_concat_fast/sample_a.wav",
                f"tos://{TOS_TEST_DIR}/audio_concat_fast/sample_b.wav",
            ],
        ],
        "output_path": [f"tos://{TOS_TEST_DIR}/audio_concat_fast/output/concatenated_audio_fast.wav"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to concatenate audio files (fast, no re-encoding)
    constructor_kwargs = {
        "output_format": "wav",  # 与输入格式一致
    }

    ds = ds.with_column(
        "output_path",
        las_udf(AudioConcatFast, construct_args=constructor_kwargs)(col("audio_paths"), col("output_path")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────╮
    # │ audio_paths                                                  ┆ output_path                                              │
    # │ ---                                                          ┆ ---                                                      │
    # │ List[Utf8]                                                   ┆ Utf8                                                     │
    # ╞══════════════════════════════════════════════════════════════╪══════════════════════════════════════════════════════════╡
    # │ [tos://tos_bucket/audio_concat_fast/sample_a.wav, ...]      ┆ tos://tos_bucket/audio_concat_fast/output/concatena... │
    # ╰──────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────╯
