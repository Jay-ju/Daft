from __future__ import annotations

import os

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioSpeakerDiarization
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    model_path = os.getenv("MODEL_PATH", "./models")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_speaker_diarization/sample.wav"]}

    input = pd.DataFrame(samples)
    df = daft.from_pandas(input)
    df = df.with_column(
        "audio_speak_diarize",
        las_udf(AudioSpeakerDiarization, construct_args={"model_path": model_path})(col("audio_path")),
    )

    df.show()
    # ╭────────────────────────────────┬───────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ audio_speak_diarize                                       │
    # │ ---                            ┆ ---                                                       │
    # │ Utf8                           ┆ List[Struct[end: Float64, speaker: Utf8, start: Float64]] │
    # ╞════════════════════════════════╪═══════════════════════════════════════════════════════════╡
    # │ tos://las-ai-cn-beijing/qa/op… ┆ [{end: 5.228,                                             │
    # │                                ┆ speaker: SPEAKE…                                          │
    # ╰────────────────────────────────┴───────────────────────────────────────────────────────────╯
