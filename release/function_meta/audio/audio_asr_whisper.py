from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_asr_whisper import AudioAsrWhisper
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_asr_whisper/参观八达岭长城。.wav"]}

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "openai/whisper-large-v3"
    audio_src_type = "audio_url"
    dtype = "bfloat16"
    source_language = "chinese"
    translate_to_english = False
    condition_on_prev_tokens = True
    compression_ratio_threshold = 1.35
    temperature = 0.5
    logprob_threshold = -1.0
    is_flat = True
    batch_size = 1
    rank = 0

    df = daft.from_pydict(samples)
    df = df.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrWhisper,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "source_language": source_language,
                "translate_to_english": translate_to_english,
                "condition_on_prev_tokens": condition_on_prev_tokens,
                "compression_ratio_threshold": compression_ratio_threshold,
                "temperature": temperature,
                "logprob_threshold": logprob_threshold,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=1,
        )(col("audio_path")),
    )
    df.show()

    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ asr_result_detail                                           │
    # │ ---                            ┆ ---                                                         │
    # │ Utf8                           ┆ Struct[asr_result: Utf8, timestamps: List[List[Float32]],   │
    # │                                ┆ segments: List[Utf8]]                                       │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/qa/op…        ┆ {asr_result: 参观八道岭长城,                                  │
    # │                                ┆ timesta…                                                    │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────────╯
