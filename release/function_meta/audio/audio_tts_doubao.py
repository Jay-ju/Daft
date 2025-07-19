from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_tts_doubao import AudioTtsDoubao
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR_URL = os.getenv("TOS_TEST_DIR_URL", "tos_bucket.tos-cn-beijing.volces.com")
    appid = os.getenv("OPENSPEECH_APPID")
    token = os.getenv("OPENSPEECH_TOKEN")

    samples = {
        "text_input": [
            "今天天气真好，适合出去走走。",
        ]
    }

    df = daft.from_pydict(samples)
    df = df.with_column(
        "tts_audio",
        las_udf(
            AudioTtsDoubao,
            construct_args={
                "appid": appid,
                "token": token,
                "uid": "test",
                "concurrency": 1,
                "timeout": 60,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text_input")),
    )

    df.show()
    # ╭─────────────────────────────────────────────┬────────────────────────────────╮
    # │ text_input                                  ┆ tts_audio                      │
    # │ ---                                         ┆ ---                            │
    # │ Utf8                                        ┆ Binary                         │
    # ╞═════════════════════════════════════════════╪════════════════════════════════╡
    # │ 欢迎使用豆包语音大模型，这是一个演示示例。… ┆ b"\xff\xf3\xe4\xc4\x00\x00\x0… │
    # ╰─────────────────────────────────────────────┴────────────────────────────────╯
