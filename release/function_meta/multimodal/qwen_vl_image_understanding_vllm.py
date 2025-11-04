from __future__ import annotations

import logging
import os

import ray

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_image_understanding_vllm import QwenVLImageUnderstandingVLLM
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
        "image_path": [
            "https://las-ai-qa-online.tos-cn-beijing.volces.com/operator_cards_serving/public/qa/shared_image_dataset/cat_ip_adapter.jpeg"
        ],
        "prompt": ["请给出图片的详细描述。"],
    }

    image_src_type = "image_url"
    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    dtype = "bfloat16"
    default_prompt = None
    max_caption_length = 256
    resized_height = None
    resized_width = None
    batch_size = 2
    seed = 42
    max_model_len = 128000
    max_num_seqs = 128
    tensor_parallel_size = 1
    enable_prefix_caching = True
    gpu_memory_utilization = 0.95
    enforce_eager = True

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenVLImageUnderstandingVLLM,
            construct_args={
                "image_src_type": image_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "prompt": default_prompt,
                "max_caption_length": max_caption_length,
                "resized_height": resized_height,
                "resized_width": resized_width,
                "batch_size": batch_size,
                "seed": seed,
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "tensor_parallel_size": tensor_parallel_size,
                "enable_prefix_caching": enable_prefix_caching,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": enforce_eager,
            },
            num_gpus=tensor_parallel_size,
            batch_size=batch_size,
            concurrency=1,
        )(col("image_path"), col("prompt")),
    )

    ds.show()

    # ╭────────────────────────────────┬─────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ image_path                     ┆ prompt                  ┆ caption                                                     │
    # │ ---                            ┆ ---                     ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8                    ┆ Utf8                                                        │
    # ╞════════════════════════════════╪═════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ 请给出图片的详细描述。…     ┆ 这张图片展示了一只拟人化的猫，它穿着一套复古风格的服装，，包括一件蓝色… │
    # ╰────────────────────────────────┴─────────────────────────┴─────────────────────────────────────────────────────────────╯
