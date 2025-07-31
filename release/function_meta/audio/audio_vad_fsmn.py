from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_vad_fsmn import AudioVadFsmn
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_vad_fsmn/参观八达岭长城。.wav"]}

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    audio_src_type = "audio_url"
    batch_size_s = 3600
    model_revision = "v2.0.4"
    rank = 0

    df = daft.from_pydict(samples)
    df = df.with_column(
        "audio_vad_result",
        las_udf(
            AudioVadFsmn,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "model_revision": model_revision,
                "batch_size_s": batch_size_s,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )
    df.show()
    df = df.with_column(
        "audio_vad_result_length",
        col("audio_vad_result").apply(
            lambda x: len(x),
            return_dtype=daft.DataType.int8(),
        ),
    )
    df.show()

    # ╭────────────────────────────────┬─────────────────────╮
    # │ audio_path                     ┆ audio_vad_result    │
    # │ ---                            ┆ ---                 │
    # │ Utf8                           ┆ List[List[Float32]] │
    # ╞════════════════════════════════╪═════════════════════╡
    # │ tos://tos_bucket/audio_asr_wh… ┆ [[0.51, 2.8]]       │
    # ╰────────────────────────────────┴─────────────────────╯
