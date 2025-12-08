from __future__ import annotations

import logging
import os

import ray

import daft
from daft import col
from daft.las.functions.audio.audio_asr_lid_whisper import AudioAsrLidWhisper
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    if os.getenv("DAFT_RUNNER", "ray") == "ray":

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

        import ray

        ray.init(dashboard_host="0.0.0.0", runtime_env={"worker_process_setup_hook": configure_logging})
        daft.context.set_runner_ray()

    daft.set_execution_config(actor_udf_ready_timeout=600)
    daft.set_execution_config(min_cpu_per_task=0)

    samples = {
        "audio_path": [
            "https://las-ai-qa-online.tos-cn-beijing.volces.com/operator_cards_serving/public/qa/shared_audio_dataset/sample_normal.wav"
        ]
    }

    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    model_name = "openai/whisper-large-v3"
    audio_src_type = "audio_url"
    punc_model_name = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
    num_gpus = 1
    device = "cuda" if num_gpus > 0 else "cpu"
    batch_size = 1
    return_language_only = False

    df = daft.from_pydict(samples)
    df = df.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrLidWhisper,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "punc_model_name": punc_model_name,
                "return_language_only": return_language_only,
                "batch_size": batch_size,
                "device": device,
            },
            num_gpus=num_gpus,
            batch_size=1,
            num_cpus=4,
            concurrency=1,
        )(col("audio_path")),
    )

    df.show()

    # ╭────────────────────────────────┬──────────────────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ asr_result_detail                                                    │
    # │ ---                            ┆ ---                                                                  │
    # │ Utf8                           ┆ Struct[asr_result: Utf8, language: Utf8, asr_result_with_punc: Utf8] │
    # ╞════════════════════════════════╪══════════════════════════════════════════════════════════════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ {asr_result: 人我保住了金我取到了俺老孙啥功名…                       │
    # ╰────────────────────────────────┴──────────────────────────────────────────────────────────────────────╯
