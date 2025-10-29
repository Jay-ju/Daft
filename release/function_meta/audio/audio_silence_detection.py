from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioSilenceDetection
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [
            f"tos://{TOS_TEST_DIR}/audio_silence_detection/sample.wav",
            f"tos://{TOS_TEST_DIR}/audio_silence_detection/silence.wav",
        ],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to detect silence in audio files
    constructor_kwargs = {
        "silence_threshold_db": -60.0,
        "timeout": 30,
    }

    ds = ds.with_column(
        "is_silence",
        las_udf(AudioSilenceDetection, construct_args=constructor_kwargs)(col("input_path")),
    )

    ds.show()
    # ╭────────────────────────────────┬────────────╮
    # │ input_path                     ┆ is_silence │
    # │ ---                            ┆ ---        │
    # │ Utf8                           ┆ Boolean    │
    # ╞════════════════════════════════╪════════════╡
    # │ tos://las-ai-qa-online/qa/tes… ┆ false      │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ tos://las-ai-qa-online/qa/tes… ┆ true       │
    # ╰────────────────────────────────┴────────────╯
