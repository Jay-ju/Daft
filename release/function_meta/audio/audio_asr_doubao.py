from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_asr_doubao import AudioAsrDoubao
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    TOS_TEST_DIR_URL = os.getenv("TOS_TEST_DIR_URL", "tos_bucket.tos-cn-beijing.volces.com")
    appid = os.getenv("OPENSPEECH_APPID")
    token = os.getenv("OPENSPEECH_TOKEN")
    samples = {"audio_path": [f"https://{TOS_TEST_DIR_URL}/audio_asr_doubao/参观八达岭长城。.wav"]}

    df = daft.from_pydict(samples)
    df = df.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrDoubao,
            construct_args={"appid": appid, "token": token, "concurrency": 1, "poll_interval": 15},
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )
    df.show()

    # ╭────────────────────────────────┬───────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ asr_result_detail                                     │
    # │ ---                            ┆ ---                                                   │
    # │ Utf8                           ┆ Struct[asr_result_raw: Utf8, asr_result_simple: Utf8] │
    # ╞════════════════════════════════╪═══════════════════════════════════════════════════════╡
    # │ https://las-ai-cn-beijing.tos… ┆ {asr_result_raw: {"audio_info…                        │
    # ╰────────────────────────────────┴───────────────────────────────────────────────────────╯
