from __future__ import annotations

import logging
import os

import ray

import daft
from daft import col
from daft.las.functions.audio.audio_denoise import AudioDenoise
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


configure_logging()

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [f"tos://{TOS_TEST_DIR}/audio/黑神话悟空对话.mp3"],
        "output_path": [f"tos://{TOS_TEST_DIR}/audio/outputs/audio_denoised/黑神话悟空对话_denoised.mp3"],
    }
    # 输出至output_path需要设置tos access_key和secret_key等认证信息
    # os.environ["TOS_ACCESS_KEY"] = os.getenv("TOS_ACCESS_KEY", "aksk")
    # os.environ["TOS_SECRET_KEY"] = os.getenv("TOS_SECRET_KEY", "aksk")
    # os.environ["TOS_ENDPOINT"] = os.getenv("TOS_ENDPOINT", "https://tos-cn-beijing.ivolces.com")
    # os.environ["TOS_REGION"] = os.getenv("TOS_REGION", "cn-beijing")

    ray.init(dashboard_host="0.0.0.0", runtime_env={"worker_process_setup_hook": configure_logging})
    daft.context.set_runner_ray()

    daft.set_execution_config(actor_udf_ready_timeout=600)
    daft.set_execution_config(min_cpu_per_task=0)

    df_samples = daft.from_pydict(samples)

    df = df_samples.with_column(
        "result_path",
        las_udf(
            AudioDenoise,
            construct_args={
                "model_path": "/opt/las/models",
            },
            num_gpus=1,
            concurrency=1,
            batch_size=2,
        )(col("audio_path"), col("output_path")),
    )

    df.show(max_width=120, format="grid")

    # ┌────────────────────────────────────┬───────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
    # │ audio_path                         │ output_path                                                       │ result_path                                                        │
    # ╞════════════════════════════════════╪═══════════════════════════════════════════════════════════════════╪════════════════════════════════════════════════════════════════════╡
    # │ tos://xxxxx/audio/黑神话悟空对话.mp3 │ tos://xxxxx/audio/outputs/audio_denoised/黑神话悟空对话_denoised.mp3 │ tos://xxxxx/audio/outputs/audio_denoised/黑神话悟空对话_denoised.mp3 │
    # └────────────────────────────────────┴───────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
