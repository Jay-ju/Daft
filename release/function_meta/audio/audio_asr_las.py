from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_asr_las import LasAsrPoller, LasAsrSubmitter
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR_URL = os.getenv("TOS_TEST_DIR_URL", "tos_bucket.tos-cn-beijing.volces.com")
    las_api_key = os.getenv("LAS_API_KEY")
    endpoint = os.getenv("LAS_SERVICE_ENDPOINT")
    samples = {"audio_path": [f"https://{TOS_TEST_DIR_URL}/audio_asr_doubao/参观八达岭长城。.wav"]}

    df = daft.from_pydict(samples)
    df = df.with_column(
        "task_id",
        las_udf(
            LasAsrSubmitter,
            construct_args={
                "api_key": las_api_key,
                "max_retries": 10,
                "endpoint": endpoint,
            },
            num_cpus=1,
            concurrency=1,
            batch_size=2,
        )(col("audio_path")),
    )

    df = df.with_column(
        "asr_result",
        las_udf(
            LasAsrPoller,
            construct_args={
                "api_key": las_api_key,
                "endpoint": endpoint,
            },
            num_cpus=1,
            concurrency=1,
            batch_size=2,
        )(col("audio_path"), col("task_id")),
    ).exclude("task_id")

    df = df.with_columns(
        {
            "asr_result_raw": col("asr_result").struct.get("asr_result_raw"),
            "asr_result_text": col("asr_result").struct.get("asr_result_text"),
            "failed_reason": col("asr_result").struct.get("failed_reason"),
        }
    ).exclude("asr_result")

    df.show()
    # ╭────────────────────────────────┬────────────────────────────────┬─────────────────┬───────────────╮
    # │ audio_path                     ┆ asr_result_raw                 ┆ asr_result_text ┆ failed_reason │
    # │ ---                            ┆ ---                            ┆ ---             ┆ ---           │
    # │ Utf8                           ┆ Utf8                           ┆ Utf8            ┆ Utf8          │
    # ╞════════════════════════════════╪════════════════════════════════╪═════════════════╪═══════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ {"audio_info":{"duration":357… ┆ 参观八达岭长城    ┆               │
    # ╰────────────────────────────────┴────────────────────────────────┴─────────────────┴───────────────╯
