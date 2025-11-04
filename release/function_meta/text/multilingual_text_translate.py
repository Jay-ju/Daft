from __future__ import annotations

import logging
import os

import ray

import daft
from daft import col
from daft.las.functions.text.multilingual_text_translate import MultilingualTextTranslate
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    os.environ["DAFT_RUNNER"] = "ray"

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

    ray.init(dashboard_host="0.0.0.0", runtime_env={"worker_process_setup_hook": configure_logging})
    daft.context.set_runner_ray()
    daft.set_execution_config(actor_udf_ready_timeout=600)
    daft.set_execution_config(min_cpu_per_task=0)

    samples = {
        "text": [
            "这是一篇关于人工智能技术发展的高质量学术论文，内容详实且具有很强的参考价值。",
        ]
    }

    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    model_name = "Seed-X-PPO-7B"
    max_model_len = 32768
    max_num_seqs = 128
    tensor_parallel_size = 1
    dtype = "bfloat16"
    enable_prefix_caching = True
    gpu_memory_utilization = 0.9
    use_cot = False
    source_language = "Chinese"
    target_language = "English"
    max_tokens = 1024
    batch_size = 4
    seed = 42

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "translate_text",
        las_udf(
            MultilingualTextTranslate,
            construct_args={
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "tensor_parallel_size": tensor_parallel_size,
                "enable_prefix_caching": enable_prefix_caching,
                "gpu_memory_utilization": gpu_memory_utilization,
                "use_cot": use_cot,
                "source_language": source_language,
                "target_language": target_language,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "seed": seed,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    ds.show()

    # ╭─────────────────────────────────────────────────────────────┬────────────────────────────────╮
    # │ text                                                        ┆ translate_text                 │
    # │ ---                                                         ┆ ---                            │
    # │ Utf8                                                        ┆ Utf8                           │
    # ╞═════════════════════════════════════════════════════════════╪════════════════════════════════╡
    # │ 这是一篇关于人工智能技术发展的高质量学术论文，内容详实且具…           ┆ This is a high-quality academ… │
    # ╰─────────────────────────────────────────────────────────────┴────────────────────────────────╯
