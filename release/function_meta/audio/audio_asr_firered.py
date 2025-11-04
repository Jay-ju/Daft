from __future__ import annotations

import logging
import os

import daft
from daft import col
from daft.las.functions.audio.audio_asr_firered import AudioAsrFireRed
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
    audio_src_type = "audio_url"
    beam_size = 3
    decode_min_len = 0
    decode_max_len = 100
    nbest = 1
    softmax_smoothing = 1.25
    aed_length_penalty = 0.6
    eos_penalty = 1.0
    repetition_penalty = 3.0
    llm_length_penalty = 1.0
    temperature = 1.0
    use_fp16 = True
    batch_size = 1
    num_gpus = 1

    df = daft.from_pydict(samples)

    # 每个UDF都需要1个GPU
    df = df.with_column(
        "asr_result_aed",
        las_udf(
            AudioAsrFireRed,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": "FireRedAsr/FireRedASR-AED-L",
                "batch_size": batch_size,
                "beam_size": beam_size,
                "decode_min_len": decode_min_len,
                "decode_max_len": decode_max_len,
                "nbest": nbest,
                "softmax_smoothing": softmax_smoothing,
                "aed_length_penalty": aed_length_penalty,
                "eos_penalty": eos_penalty,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    df = df.with_column(
        "asr_result_llm",
        las_udf(
            AudioAsrFireRed,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": "FireRedAsr/FireRedASR-LLM-L",
                "batch_size": batch_size,
                "beam_size": beam_size,
                "decode_min_len": decode_min_len,
                "decode_max_len": decode_max_len,
                "repetition_penalty": repetition_penalty,
                "llm_length_penalty": llm_length_penalty,
                "use_fp16": use_fp16,
                "temperature": temperature,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )
    df.show()

    # ╭────────────────────────────────┬────────────────────────────────────────────┬───────────────────────────────────────────╮
    # │ audio_path                     ┆ asr_result_aed                             ┆ asr_result_llm                            │
    # │ ---                            ┆ ---                                        ┆ ---                                       │
    # │ Utf8                           ┆ Utf8                                       ┆ Utf8                                      │
    # ╞════════════════════════════════╪════════════════════════════════════════════╪═══════════════════════════════════════════╡
    # │ https://las-ai-qa-online.to…   ┆ 人我保住了金我取到了俺老孙啥功名不要只求回        ┆ 人我保住了金我取到了俺老孙啥功名不要只求         │
    # │                                ┆ 到这花果山终老过…                             ┆ 回到这花果山中了过…                          │
    # ╰────────────────────────────────┴────────────────────────────────────────────┴───────────────────────────────────────────╯
