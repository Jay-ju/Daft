from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_lid_whisper import AudioLidWhisper
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_lid_whisper/参观八达岭长城。.wav"]}

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "iic/speech_whisper-large_lid_multilingual_pytorch"
    model_version = "v2.0.4"
    num_gpus = 1
    rank = 0

    df = daft.from_pydict(samples)
    df = df.with_column(
        "lid_result",
        las_udf(
            AudioLidWhisper,
            construct_args={
                "model_path": model_path,
                "model_name": model_name,
                "model_version": model_version,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )
    df.show()

    # ╭────────────────────────────────┬────────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ lid_result                                                 │
    # │ ---                            ┆ ---                                                        │
    # │ Utf8                           ┆ Struct[language_code: Utf8, language_code_full_name: Utf8] │
    # ╞════════════════════════════════╪════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_lid_wh… ┆ {language_code: zh,                                        │
    # │                                ┆ language_…                                                 │
    # ╰────────────────────────────────┴────────────────────────────────────────────────────────────╯
