from __future__ import annotations

import logging
import os

import ray

import daft
from daft import col
from daft.las.functions.audio.audio_asr_las import LasAsrPoller, LasAsrSubmitter
from daft.las.functions.udf import las_udf


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S.%s".format(),
    )
    logging.getLogger("tracing.span").setLevel(logging.WARNING)
    logging.getLogger("daft_io.stats").setLevel(logging.WARNING)
    logging.getLogger("DaftStatisticsManager").setLevel(logging.WARNING)
    logging.getLogger("DaftFlotillaScheduler").setLevel(logging.WARNING)
    logging.getLogger("DaftFlotillaDispatcher").setLevel(logging.WARNING)


if __name__ == "__main__":
    TOS_TEST_DIR_URL = os.getenv("TOS_TEST_DIR_URL", "tos_bucket.tos-cn-beijing.volces.com")
    las_api_key = os.getenv("LAS_API_KEY")
    endpoint = os.getenv("LAS_SERVICE_ENDPOINT")
    samples = {"audio_path": [f"https://{TOS_TEST_DIR_URL}/audio_asr_doubao/参观八达岭长城。.wav"]}

    ray.init(dashboard_host="0.0.0.0", runtime_env={"worker_process_setup_hook": configure_logging})
    daft.context.set_runner_ray()

    daft.set_execution_config(actor_udf_ready_timeout=600)
    daft.set_execution_config(min_cpu_per_task=0)

    df_samples = daft.from_pydict(samples)
    df = df_samples.with_column(
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
            "asr_result_simple": col("asr_result").struct.get("asr_result_simple"),
            "failed_reason": col("asr_result").struct.get("failed_reason"),
        }
    ).exclude("asr_result")

    df.show()
    # ╭────────────────────────────────┬────────────────────────────────┬──────────────────┬───────────────────────────────────┬───────────────╮
    # │ audio_path                     ┆ asr_result_raw                 ┆ asr_result_text  ┆ asr_result_simple                 ┆ failed_reason │
    # │ ---                            ┆ ---                            ┆ ---              ┆ ---                               ┆ ---           │
    # │ Utf8                           ┆ Utf8                           ┆ Utf8             ┆ Utf8                              ┆ Utf8          │
    # ╞════════════════════════════════╪════════════════════════════════╪══════════════════╪═══════════════════════════════════╪═══════════════╡
    # │ https://las-ai-cn-beijing.tos… ┆ {"additions": {"duration": "3… ┆ 参观八达岭长城。    ┆ 0:00:00 0:00:02 参观八达岭长城。…    ┆               │
    # ╰────────────────────────────────┴────────────────────────────────┴──────────────────┴───────────────────────────────────┴───────────────╯

    # Add language column and enable speaker info
    df = df_samples.with_column(
        "language",
        daft.lit({"language": "", "format": "wav"}),
        # When this key is empty, the model supports recognition of Chinese, English, Shanghainese, Minnan, Sichuan, Shaanxi, and Cantonese.
        # When it is set to a specific key below, it can recognize the specified language.
        # English: en-US
        # Japanese: ja-JP
        # Indonesian: id-ID
        # Spanish: es-MX
        # Portuguese: pt-BR
        # German: de-DE
        # French: fr-FR
        # Korean: ko-KR
        # Filipino: fil-PH
        # Malay: ms-MY
        # Thai: th-TH
        # Arabic: ar-SA
        # For example, if the input audio is German, this parameter should be 'de-DE'.
    )

    df = df.with_column(
        "task_id",
        las_udf(
            LasAsrSubmitter,
            construct_args={
                "api_key": las_api_key,
                "max_retries": 10,
                "endpoint": endpoint,
                "enable_speaker_info": True,
            },
            num_cpus=1,
            concurrency=1,
            batch_size=2,
        )(col("audio_path"), col("language")),
    )

    df = df.with_column(
        "asr_result",
        las_udf(
            LasAsrPoller,
            construct_args={
                "api_key": las_api_key,
                "endpoint": endpoint,
                "enable_speaker_info": True,
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
            "asr_result_simple": col("asr_result").struct.get("asr_result_simple"),
            "failed_reason": col("asr_result").struct.get("failed_reason"),
        }
    ).exclude("asr_result")

    df.show()

    # ╭────────────────────────────────┬──────────────────────────────────────┬────────────────────────────────┬──────────────────┬──────────────────────────┬───────────────╮
    # │ audio_path                     ┆ language                             ┆ asr_result_raw                 ┆ asr_result_text  ┆ asr_result_simple        ┆ failed_reason │
    # │ ---                            ┆ ---                                  ┆ ---                            ┆ ---              ┆ ---                      ┆ ---           │
    # │ Utf8                           ┆ Struct[language: Utf8, format: Utf8] ┆ Utf8                           ┆ Utf8             ┆ Utf8                     ┆ Utf8          │
    # ╞════════════════════════════════╪══════════════════════════════════════╪════════════════════════════════╪══════════════════╪══════════════════════════╪═══════════════╡
    # │ https://las-ai-cn-beijing.tos… ┆ {language: ,                         ┆ {"additions": {"duration": "3… ┆ 参观八达岭长城。 ┆ 说话人 1 0:00:00 0:00:02 ┆               │
    # │                                ┆ format: wav,                         ┆                                ┆                  ┆ 参观八达岭长城…          ┆               │
    # │                                ┆ }                                    ┆                                ┆                  ┆                          ┆               │
    # ╰────────────────────────────────┴──────────────────────────────────────┴────────────────────────────────┴──────────────────┴──────────────────────────┴───────────────╯
