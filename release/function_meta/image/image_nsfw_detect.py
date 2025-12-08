from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.image.image_nsfw_detect import ImageNsfwDetect
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    if os.getenv("DAFT_RUNNER", "native") == "ray":
        import logging

        import ray

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
        "image": [
            "https://las-ai-qa-online.tos-cn-beijing.volces.com/operator_cards_serving/public/qa/shared_image_dataset/cat_ip_adapter.jpeg"
        ]
    }

    image_src_type = "image_url"
    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    model_name = "Falconsai/nsfw_image_detection"
    rank = 0
    num_gpus = 0
    batch_size = 1

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "nsfw_detect",
        las_udf(
            ImageNsfwDetect,
            construct_args={
                "image_src_type": image_src_type,
                "batch_size": batch_size,
                "model_path": model_path,
                "model_name": model_name,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image")),
    )

    ds.show()

    # ╭────────────────────────────────┬────────────────────────╮
    # │ image                          ┆ nsfw_detect            │
    # │ ---                            ┆ ---                    │
    # │ Utf8                           ┆ Float64                │
    # ╞════════════════════════════════╪════════════════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ 0.000114               │
    # ╰────────────────────────────────┴────────────────────────╯
